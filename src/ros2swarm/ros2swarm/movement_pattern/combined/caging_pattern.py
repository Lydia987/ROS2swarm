from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image
from communication_interfaces.msg import StringMessage
from rclpy.qos import qos_profile_sensor_data
from ros2swarm.utils import setup_node
from ros2swarm.utils.state import State
from ros2swarm.movement_pattern.movement_pattern import MovementPattern
from ros2swarm.utils.scan_calculation_functions import ScanCalculationFunctions

from scipy.special import expit
import numpy as np
import numdifftools as nd
import time
from threading import Timer
import imutils
import cv2 as cv
from cv_bridge import CvBridge

bridge = CvBridge()


# TODO: Wieso aufhänger, wenn ganze laser nachricht gespeichert wird
# TODO: Schwerpunkt von concave noch richtig setzten
# TODO: Fehleranfälligkeit bei zu vielen Robotern reduzieren durch raushalten eines Roboters aus dem Transport,
#        wenn schon mehr als 2 Roboter als benötigt im quorum sind
# TODO: D_min und D_max muss noch selbständig abgeschätzt werden und vermutlich auch e
# TODO: abschätzung der höhe, möglich bei approach mehrere messungen von dist und daraus abschätzung der höhe im kamerabild
# TODO: Es wird near_object falsch aktualisiert, weshalb zuerst in surround gegangen wird
# TODO: Positionen noch anpassen (Parameter l beachten!!!)
# TODO: Vektorfeld noch anpassen, sollte bei großem K beim Kreis äußeren direkt richtung kreis zeigen und beim kreis inneren direkt nach draußen!
#       * eine Version davon ist schon vorhanden,
#           allerdings bleiben die roboter nahe zu stehen wenn sie in richtung des Kreises kommen
#           und dies wird nicht durch psi behoben
#       * offensichtlich wird psi irgendwie noch falsch berechnet
# TODO: Eventuell extra Nachrichten typ für quorum_msg anlegen
# IDEE: da die roboter das Objekt immer umkreisen sollten sie nach jeden umkreisen das ziel mindestens einmal gesehen haben
#       roboter merkt sich relative position zum ziel durch Odometrie bis er es wieder gesehen hat
#       Gewichtung der shape control um so stärker, je länger er das ziel nicht sieht
#       bis zur ersten hälfte der umrundung größeres vertrauen in hintermann und danach in vordermann


def get_neighbors(scan_msg, max_range):
    # TODO: Params für den threshold und width abhängig von roboter typ anlegen
    # center: erster wert entfernung in meter , zweiter wert rad (vorne 0, links rum steigend bis 2pi)
    # roboter werden zwischen 0.2m und 3.5m erkannt min_range=0, max_range=3.5, threshold=0.35, min_width=0, max_width=15
    robots, robots_center = ScanCalculationFunctions.identify_robots(laser_scan=scan_msg, min_range=0,
                                                                     max_range=max_range, threshold=0.35, min_width=0,
                                                                     max_width=15)
    return robots, robots_center


def get_centroid(img):
    x = 0
    y = 0

    # Nur damit Kontur in anderer Farbe eingezeichnet werden kann TODO: wieder entfernen
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    median = cv.medianBlur(grey, 5)
    blurred = cv.GaussianBlur(median, (5, 5), 0)
    thresh = cv.threshold(blurred, 60, 255, cv.THRESH_BINARY)[1]

    # find contours in the threshold image
    contours = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    it = 0
    # loop over the contours
    for c in contours:
        it += 1
        # compute the center of the contour
        M = cv.moments(c)
        if M["m00"] != 0:
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
        # draw the contour and center of the shape on the image TODO: wieder entfernen
        cv.drawContours(img, [c], -1, (255, 0, 0), 2)
        cv.circle(img, (x, y), 7, (255, 0, 0), -1)
        cv.putText(img, str(it), (x - 50, y - 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # speichert Bild TODO: wieder entfernen
    cv.imwrite('Thresh.jpeg', thresh)
    cv.imwrite('Center.jpeg', img)

    return x, y


def get_distance(scan_msg, degree):
    rad = np.deg2rad(degree)
    increment = scan_msg.angle_increment
    min_rad = scan_msg.angle_min
    max_rad = scan_msg.angle_max
    if rad < min_rad or rad > max_rad:
        distance = np.inf
    else:
        distance = scan_msg.ranges[int((rad - min_rad) / increment)]
    return distance


def is_color_in_image(image, lower_color, upper_color):
    is_in_image = False
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    obj_mask = cv.inRange(hsv, lower_color, upper_color)
    number_of_white_pix = np.sum(obj_mask == 255)
    number_of_pix = obj_mask.shape[0] * obj_mask.shape[1]
    percentage_of_white_pix = number_of_white_pix * (100 / number_of_pix)
    if percentage_of_white_pix > 0.01:
        is_in_image = True
    return is_in_image


def sigma_plus(omega):
    return 1 / (1 + expit(1 - omega))


def sigma_minus(omega):
    return 1 / (1 + expit(omega - 1))


def f(N):
    return sigma_plus(N)


def g(T):
    return 1 - sigma_minus(T)


def s(x, y, r, center):
    # s(x, y) = 0 with s(x, y) < 0 for (x, y) inside ∂S and s(x, y) > 0 for (x, y) outside ∂S
    """
    Signed distance function for a circle with center (cx, cy) and radius r.
    Returns a 2D array of signed distances, evaluated at the points (x, y).
    """
    # Eigentlich richtige Funktion, die aber aufgrund von anderen fehlern aktuell noch dazu führt das die Roboter stehen bleiben
    # TODO: wieder einkommentieren wenn der Fehler behoben wurde
    # cx = center[0]
    # cy = center[1]
    #
    # # Compute the Euclidean distance between each point (x,y) and the center (cx,cy)
    # d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    #
    # # Compute the signed distance function
    # shape = d - r
    #
    # # Set the value to negative inside the circle
    # if d < r:
    #     shape = -shape

    # TODO: wieder auskommentieren wenn der Fehler behoben wurde
    circle_function = (x - center[0]) ** 2 + (y - center[1]) ** 2
    shape = circle_function - r ** 2
    # if circle_function - r ** 2 < 0:
    #     shape = -shape
    return shape


def get_trajectory(start, goal, distance):
    # Calculate the vector from start to goal
    vector = np.array(goal) - np.array(start)

    # Calculate the length of the vector
    length = np.linalg.norm(vector)

    # Calculate the unit vector in the direction of the vector
    unit_vector = vector / length

    # Calculate the scaled vector with length equal to `distance`
    scaled_vector = distance * unit_vector

    # Calculate the final point by adding the scaled vector to the start point
    point = np.array(start) + scaled_vector

    return tuple(point)


def get_mean_dist(scan_msg, min_angle, max_angle):
    """ calculates the mean distance to obstacles between the min_angle and max_angle """
    R = scan_msg.range_max  # repulsive force up to R in meter
    sum_dist = 0.0
    start_angle = np.deg2rad(min_angle)  # rad (front = 0 rad)
    end_angle = np.deg2rad(max_angle)  # rad (front = 0 rad)

    start = int((start_angle - scan_msg.angle_min) / scan_msg.angle_increment)
    end = int((end_angle - scan_msg.angle_min) / scan_msg.angle_increment)

    if min_angle == max_angle:
        return scan_msg.ranges[start]

    for i in range(start, end):
        dist_i = scan_msg.ranges[i]
        if np.isnan(dist_i):
            dist_i = 0.0
        if np.isinf(dist_i) or dist_i > R:
            dist_i = R

        sum_dist += dist_i

    mean_dist = sum_dist / (end - start)
    return mean_dist


class CagingPattern(MovementPattern):
    """
    Pattern to cage an object and transport it to the target.
    """

    def __init__(self):
        """Initialize the caging pattern node."""
        super().__init__('caging_pattern')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('turn_timer_period', None),
                ('goal_search_timer_period', None),
                ('object_search_timer_period', None),
                ('object_distance_reached', None),
                ('max_translational_velocity', None),
                ('max_rotational_velocity', None),
                ('robot_radius', None),
                ('robot_length', None)
            ])

        # Params
        self.param_max_translational_velocity = self.get_parameter(
            "max_translational_velocity").get_parameter_value().double_value
        self.param_max_rotational_velocity = self.get_parameter(
            "max_rotational_velocity").get_parameter_value().double_value
        self.param_distance_reached = self.get_parameter("object_distance_reached").get_parameter_value().double_value
        self.param_turn_timer_period = self.get_parameter("turn_timer_period").get_parameter_value().double_value
        self.param_goal_timer_period = self.get_parameter("goal_search_timer_period").get_parameter_value().double_value
        self.param_object_timer_period = self.get_parameter(
            "object_search_timer_period").get_parameter_value().double_value
        self.r = self.get_parameter("robot_radius").get_parameter_value().double_value
        self.l = self.get_parameter("robot_length").get_parameter_value().double_value

        # messages from subpattern subscription
        self.random_walk_latest = Twist()
        self.random_walk_subpattern = self.create_subscription(Twist,
                                                               self.get_namespace() + '/drive_command_random_walk_pattern',
                                                               self.command_callback_random_walk, 10)
        # Subscriber
        self.model_states_subscription = self.create_subscription(ModelStates, '/gazebo/model_states',
                                                                  self.model_states_callback,
                                                                  qos_profile=qos_profile_sensor_data)
        self.odom_subscription = self.create_subscription(Odometry, self.get_namespace() + '/odom', self.odom_callback,
                                                          qos_profile=qos_profile_sensor_data)
        self.scan_subscription = self.create_subscription(
            LaserScan, self.get_namespace() + '/scan', self.swarm_command_controlled(self.scan_callback),
            qos_profile=qos_profile_sensor_data)
        self.camera_subscription = self.create_subscription(
            Image, self.get_namespace() + '/camera/image_raw', self.swarm_command_controlled(self.camera_callback),
            qos_profile=qos_profile_sensor_data)
        self.quorum_subscription = self.create_subscription(
            StringMessage, '/quorum', self.quorum_callback, 10)
        self.quorum_publisher = self.create_publisher(
            StringMessage, '/quorum', 10)
        self.goal_subscription = self.create_subscription(
            StringMessage, '/goal', self.goal_callback, 10)

        # Publisher
        self.goal_publisher = self.create_publisher(
            StringMessage, '/goal', 10)
        self.information_publisher = self.create_publisher(
            StringMessage, self.get_namespace() + '/information', 10)
        self.protection_publisher = self.create_publisher(
            StringMessage, self.get_namespace() + '/hardware_protection_layer', 10)

        self.direction = Twist()
        self.state = State.INIT
        self.publish_state_counter = 0
        self.state_switch_counter = 0
        self.quorum_switch_counter = 0
        self.last_error = 0.0
        self.integral_error = 0.0
        self.closure = False
        self.quorum = False
        self.quorum_dict = {}
        self.quorum_msg = StringMessage()
        self.quorum_msg.data = 'False'
        self.d_min = 0.0  # 2.7
        self.d_max = 0.0  # 3.2
        self.e = 0.15  # war 0.15 bei 2x2 Kreis gut, bei 3x2 quader 0.23
        self.R = 3.0  # 1.5  # Sensor_Range in m
        self.Q = [[0.0,
                   0.0]]  # Liste mit der eigenen Position an erster Stelle und danach die Positionen aller Nachbarn (alle Roboter die sich in der Range R um den Roboter befinden)
        self.clockwise = False
        self.current_pose = [0.0, 0.0, 0.0]  # [x, y, orientation] in [m, m, rad]
        self.odometry_list = []
        self.start = np.inf  # index of the odometry list, where move around object started
        self.current_scan = None
        self.current_image = None

        # color in RGB = [76, 142, 24] , in BGR = [24, 142, 76] and in HSV = [47 212 142]
        # [100, 50, 50] bis [140, 255, 255] entspricht rot, aber warum ???
        # [50, 100, 100] bis [70, 255, 255] entspricht grün
        # print(cv.cvtColor(np.uint8([[BGR]]),cv.COLOR_BGR2HSV))
        self.lower_object_color = np.array([50, 50, 50])  # np.array([50, 50, 50])  # in HSV [H-10, 100,100]
        self.upper_object_color = np.array([70, 255, 255])  # np.array([70, 255, 255])  # in HSV [H+10, 255, 255]
        self.near_object = False
        self.object_in_image = False
        self.object_direction = None
        self.object_distance = np.inf
        self.current_object_center = np.array([0.0, 0.0])
        self.object_name = "concave"  # "concave" # "Green_Quader_3x2"  # "Green"

        self.lower_goal_color = np.array([110, 100, 100])  # np.array([0, 50, 50])  # in HSV [H-10, 100,100]
        self.upper_goal_color = np.array([130, 255, 255])  # np.array([10, 255, 255])  # in HSV [H+10, 255, 255]
        self.goal_in_image = False
        self.goal_position = [0.0, 0.0]
        self.goal_name = "Red"
        self.goal_dict = {}
        self.goal_msg = StringMessage()
        self.goal_msg.data = 'False'

        self.info = StringMessage()
        self.info.data = 'Starting Caging Pattern'
        self.information_publisher.publish(self.info)

        self.protection = StringMessage()
        self.protection.data = 'True'
        self.protection_publisher.publish(self.protection)

        # Timer
        # self.state_timer = Timer(0.01, self.caging)
        # self.state_timer.start()
        self.search_goal_timer = Timer(self.param_goal_timer_period, self.is_goal_visible,
                                       args=(self.lower_goal_color, self.upper_goal_color))
        self.search_object_timer = Timer(self.param_object_timer_period, self.is_object_visible,
                                         args=(self.lower_object_color, self.upper_object_color))
        self.turn_timer = Timer(self.param_turn_timer_period, self.stop)
        self.last_call_time = time.time()

        # Variablen für das Testen und Auswerten
        self.shape = [[], []]
        self.test_counter = 200
        self.robots_in_quorum = 0
        self.error_list = []
        self.should_orientation = 0.0
        self.dist = -1.0
        # 0 = time, 1 = velocity, 2 = state, 3 = pose, 4 = object_center, 5 = error, 6 = should_orientation,
        # 7 = #robots in quorum, 8 = goal_position, 9 = quorum, 10 = closure, 11 = object_shape]
        self.state_list = [[], [[], []], [], [[], [], []], [[], []], [], [], [], [[], []], [], [], [[], []]]

    # PUBLISHER #
    def publish_state(self):
        self.state_list[0].append([time.time()])
        self.state_list[1][0].append(self.direction.linear.x)
        self.state_list[1][1].append(self.direction.angular.z)
        self.state_list[2].append([str(self.state)])
        self.state_list[3][0].append(self.current_pose[0])
        self.state_list[3][1].append(self.current_pose[1])
        self.state_list[3][2].append(np.rad2deg(self.current_pose[2]))
        self.state_list[4][0].append(self.current_object_center[0])
        self.state_list[4][1].append(self.current_object_center[1])
        self.state_list[5].append([self.last_error])
        self.state_list[6].append([self.should_orientation])
        self.state_list[7].append([self.robots_in_quorum])
        self.state_list[8][0].append(self.goal_position[0])
        self.state_list[8][1].append(self.goal_position[1])
        self.state_list[9].append([int(self.quorum)])
        self.state_list[10].append([int(self.closure)])
        # if len(self.odometry_list) > 0:
        #     self.state_list[11][0].append(self.getShape(self.odometry_list)[0])
        #     self.state_list[11][1].append(self.getShape(self.odometry_list)[1])
        if len(self.shape[0]) > 0:
            self.state_list[11][0].append(self.shape[0])
            self.state_list[11][1].append(self.shape[1])
        else:
            self.state_list[11][0].append(0.0)
            self.state_list[11][1].append(0.0)
        with open('state_list_' + str(self.get_namespace())[-1] + '.txt', 'w') as doc:
            doc.write(str(self.state_list))

        self.publish_state_counter -= 1
        if self.publish_state_counter <= 0:
            self.publish_state_counter = 10
            self.info.data = '_____' + str(self.state) + '_____ near_object=' + str(
                self.near_object) + ' object=' + str(
                self.object_in_image) + ' distance=' + str(self.object_distance) + ' quorum=' + str(
                self.quorum) + ' closure=' + str(self.closure)
            self.information_publisher.publish(self.info)

    def publish_info(self, msg):
        self.info.data = str(msg)
        self.information_publisher.publish(self.info)

    # CALLBACKS #
    def model_states_callback(self, model_states):
        index = 0
        for name in model_states.name:
            if name == 'robot_name_' + self.get_namespace()[-1]:
                self.current_pose[0] = model_states.pose[index].position.x
                self.current_pose[1] = model_states.pose[index].position.y

                z = model_states.pose[index].orientation.z
                w = model_states.pose[index].orientation.w

                if np.sign(z) < 0:  # clockwise
                    self.current_pose[2] = 2 * np.pi - 2 * np.arccos(w)
                else:  # anticlockwise
                    self.current_pose[2] = 2 * np.arccos(w)

            elif name == self.object_name:
                self.current_object_center[0] = model_states.pose[index].position.x
                self.current_object_center[1] = model_states.pose[index].position.y

            # TODO: wieder auskommentieren, wenn die Zielabschätzung besser ist
            elif name == self.goal_name:
                self.goal_position[0] = model_states.pose[index].position.x
                self.goal_position[1] = model_states.pose[index].position.y
            index += 1

    def odom_callback(self, odom_msg):
        if self.current_scan is not None and self.state == State.MOVE_AROUND_OBJECT:
            currentPosition = [odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y]
            z = odom_msg.pose.pose.orientation.z
            w = odom_msg.pose.pose.orientation.w
            if np.sign(z) < 0:  # clockwise
                currentOrientation = 2 * np.pi - 2 * np.arccos(w)
            else:  # anticlockwise
                currentOrientation = 2 * np.arccos(w)
            # objectDist = get_distance(self.current_scan, 90)
            # if abs(objectDist) > self.current_scan.range_max:
            #     objectDist = self.current_scan.range_max * np.sign(objectDist)
            # self.odometry_list.append([currentPosition, currentOrientation, objectDist])

            self.odometry_list.append([currentPosition, currentOrientation, self.current_scan])

    def quorum_callback(self, incoming_msg: StringMessage):
        name, value = incoming_msg.data.split(' ')
        self.quorum_dict[name] = value

    def goal_callback(self, incoming_msg: StringMessage):
        """Call back if a new goal msg is available."""
        name, value = incoming_msg.data.split(' ')
        self.goal_dict[name] = value

    def command_callback_random_walk(self, incoming_msg: Twist):
        """Assign the message to variable"""
        self.random_walk_latest = incoming_msg

    def scan_callback(self, incoming_msg: LaserScan):
        """Call back if a new scan msg is available."""
        # self.publish_info('scan_callback')
        self.current_scan = incoming_msg
        self.caging()

    def camera_callback(self, raw_image_msg: Image):
        """Call back if a new scan msg is available."""
        # self.publish_info('camera_callback')
        self.current_image = bridge.imgmsg_to_cv2(raw_image_msg, desired_encoding='passthrough')

    # STATUS AKTUALISIERUNG UND AUSFÜHRUNG #
    def caging(self):
        # Testprogramm, wie viel kann ein Roboter schieben
        # self.direction.linear.x = self.param_max_translational_velocity
        # self.direction.angular.z = 0.0
        # self.protection.data = 'False'
        # self.protection_publisher.publish(self.protection)
        # self.command_publisher.publish(self.direction)

        # self.publish_info('caging')
        if self.is_busy():
            return

        self.update_flags()
        self.update_state()
        self.publish_state()  # TODO: Wieder entfernen, nur zum Debuggen
        self.execute_state()

    def update_flags(self):
        # self.publish_info('update_flags')
        old_quorum = self.quorum
        old_goal_in_image = self.goal_in_image

        if self.state == State.SEARCH:  # or self.state == State.APPROACH:
            self.update_object_infos(self.current_scan, self.current_image, self.lower_object_color,
                                     self.upper_object_color)
            self.quorum = False
            self.closure = False

        if self.state == State.APPROACH or self.state == State.SURROUND:
            self.update_is_near_object()

        if self.state == State.SURROUND or self.state == State.TRANSPORT:
            self.update_Q()
            # if self.test_counter < 0:
            #     self.closure = True
            #     self.quorum = True
            # else:
            #     self.test_counter -= 1
            self.update_is_closure()
            self.update_is_quorum(self.current_scan)
            if (not self.quorum) and old_quorum and self.quorum_switch_counter > 30:
                self.quorum = True
                self.quorum_switch_counter = 0
            else:
                self.quorum_switch_counter += 1

        if self.state == State.TRANSPORT:
            self.update_goal_position()
            self.goal_in_image = is_color_in_image(self.current_image, self.lower_goal_color, self.upper_goal_color)
            if old_goal_in_image != self.goal_in_image:
                self.goal_msg.data = str(self.get_namespace()) + ' ' + self.calculate_goal_position()
                self.goal_publisher.publish(self.goal_msg)

        if old_quorum != self.quorum:
            self.quorum_msg.data = str(self.get_namespace()) + ' ' + str(self.quorum)
            self.quorum_publisher.publish(self.quorum_msg)

    def update_state(self):
        # self.publish_info('update_state')
        old_state = self.state

        if self.state != State.INIT:
            if self.state == State.SEARCH:
                if self.object_in_image:
                    if self.d_max <= 0.0:
                        self.state = State.MOVE_AROUND_OBJECT
                    else:
                        self.state = State.APPROACH

            elif self.state == State.MOVE_AROUND_OBJECT:
                if self.d_max <= 0.0:
                    self.state = State.MOVE_AROUND_OBJECT
                else:
                    self.state = State.SURROUND

            elif self.state == State.APPROACH:
                if self.near_object:
                    self.state = State.SURROUND
                #
                # if not self.object_in_image:
                #     self.state = State.SEARCH
                # elif self.near_object:
                #     self.state = State.SURROUND

            elif self.state == State.SURROUND:
                if not self.near_object:
                    self.state = State.APPROACH
                elif self.closure and self.quorum:
                    self.state = State.TRANSPORT

            elif self.state == State.TRANSPORT:
                if not (self.closure and self.quorum):
                    self.state = State.SURROUND

        if self.state != old_state and (old_state == State.SURROUND or old_state == State.TRANSPORT) \
                and self.state != State.TRANSPORT and self.state_switch_counter < 20:
            self.publish_info("state_switch_counter")
            self.state_switch_counter += 1
            self.state = old_state
        else:
            self.state_switch_counter = 0

    def execute_state(self):
        # self.publish_info('execute_state')
        # führt Verhalten abhängig von Status aus
        if self.state == State.INIT:
            fully_initialized = not (self.current_image is None or self.current_scan is None)
            if fully_initialized:
                self.search_object_timer.start()
                self.direction = self.random_walk_latest
                self.state = State.SEARCH  # State.APPROACH

        elif self.state == State.SEARCH:
            self.direction = self.random_walk_latest
            self.protection.data = 'True'

        elif self.state == State.MOVE_AROUND_OBJECT:
            self.direction = self.moveAroundObject()
            self.protection.data = 'True'

        elif self.state == State.APPROACH:  # if d_obj(q_i) > d_near object
            # self.direction.angular.z = self.object_direction
            # self.direction.linear.x = self.param_max_translational_velocity
            self.direction = self.approach()
            self.protection.data = 'True'

        elif self.state == State.SURROUND:  # if d_obj(q_i) ≤ d_near object
            self.direction = self.surround()
            self.protection.data = 'False'

        elif self.state == State.TRANSPORT:
            self.direction = self.transport()
            self.protection.data = 'False'

        self.protection_publisher.publish(self.protection)
        self.command_publisher.publish(self.direction)

    # BEWEGUNGSMUSTER #
    def approach(self):
        self.update_Q()
        v, w = self.shape_control(0.5)
        approach = Twist()
        approach.linear.x = v
        approach.angular.z = w
        return approach

    def surround(self):
        v, w = self.shape_control(0.0)
        surround = Twist()
        surround.linear.x = v
        surround.angular.z = w
        return surround

    def transport(self):
        v, w = self.shape_control(0.4)
        transport = Twist()
        transport.linear.x = v
        transport.angular.z = w
        return transport

    def shape_control(self, K):
        # K controls the rate of descent to the specified surface relative the orbiting velocity and is used to help generate different behaviors
        q = self.Q

        dx = q[0][0] - self.current_object_center[0]
        dy = q[0][1] - self.current_object_center[1]
        theta = np.arctan2(dy, dx)
        vector = [np.cos(theta) * self.gamma(q[0]), np.sin(theta) * self.gamma(q[0])]
        # vector = nd.Gradient(self.phi)(q[0])

        minute = np.multiply(np.multiply(-1.0 * K, f(self.N())), np.array(vector))
        subtrahend = np.multiply(self.gradient_cross_psi(q[0]), g(self.T()))
        u = np.subtract(minute, subtrahend)

        v = np.sqrt(u[0] ** 2 + u[1] ** 2)
        w = np.arctan2(u[1], u[0])
        self.should_orientation = np.rad2deg(w)
        w = w - self.current_pose[2]
        w = np.arctan2(np.sin(w), np.cos(w))  # calculates the minimal turn angle

        # self.error_list.append([time.time(), w])
        # with open('error_list.txt', 'w') as doc:
        #     doc.write(str(self.error_list))

        w = self.pid_controller(w)
        v = v * 0.45
        w = w * 0.45
        v, w = self.scale(v, w)

        return v, w

    def pid_controller(self, error):
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        if time_since_last_call > 0.5:
            time_since_last_call = 0.0
            self.integral_error = 0.0
            self.last_error = 0.0
        self.last_call_time = current_time

        # Ursprüngliche werte nach Ziegler und Nicholson 50, 1.78
        # Gute Werte: 30, 1.78, 0.5, 0.7, 0.3
        # letzte Werte: 20, 1.78, 0.5, 0.7, 0.25
        k_pr_crit = 20
        t_p = 1.78
        dt = time_since_last_call
        kp = 0.5 * k_pr_crit
        ki = 0.7 * t_p
        kd = 0.25 * t_p

        self.integral_error += error * dt
        derivative_error = (error - self.last_error) / dt if dt > 0 else 0
        output = kp * error + ki * self.integral_error + kd * derivative_error
        self.last_error = error
        return output

    def scale(self, translational_velocity, rotational_velocity):
        if abs(translational_velocity) > self.param_max_translational_velocity:
            scale = self.param_max_translational_velocity / abs(translational_velocity)
            translational_velocity = translational_velocity * scale
            rotational_velocity = rotational_velocity * scale
        if abs(rotational_velocity) > self.param_max_rotational_velocity:
            scale = self.param_max_rotational_velocity / abs(rotational_velocity)
            translational_velocity = translational_velocity * scale
            rotational_velocity = rotational_velocity * scale
        return translational_velocity, rotational_velocity

    def stop(self):
        stop = Twist()
        self.command_publisher.publish(stop)

    def drive_backwards(self):
        backwards, obstacle_free = ScanCalculationFunctions.potential_field(2.0, 0.7,
                                                                            self.param_max_rotational_velocity,
                                                                            self.param_max_translational_velocity,
                                                                            0.12, self.current_scan,
                                                                            5, None)
        if not obstacle_free:
            self.command_publisher.publish(backwards)
            # self.publish_info('backwards START')

            # TODO: param daraus machen der in der Yaml datei steht oder besser in abhängigkeit der max geschwindigkeit
            time.sleep(1.0)

    def turn_once(self):
        self.turn_timer = Timer(self.param_turn_timer_period, self.stop)
        turn = Twist()
        turn.angular.z = self.param_max_rotational_velocity

        self.command_publisher.publish(turn)
        self.turn_timer.start()

        self.publish_info('!!!!!!!!!!!!!!! turn_timer START !!!!!!!!!!!!!!!!!')

    # FLAGGEN #
    def is_busy(self):
        if self.turn_timer.is_alive() and self.state != State.INIT:
            return True
        else:
            return False

    def update_is_quorum(self, scan_msg):
        # TODO: mit paper überprüfen ob richtig implementiert
        # TODO: max_range sollte laut dem Paper D_min(obj) entsprechen
        max_range = 3.5  # eigentlich self.d_min + self.r vermutlich läuft es aber mit einer etwas größeren range besser
        robots, robots_center = get_neighbors(scan_msg, max_range)
        front = 0
        back = 0
        for robot in robots_center:
            if robot[1] < np.deg2rad(
                    80):  # robot[1] > np.deg2rad(280): value clockwise, robot[1] < np.deg2rad(80): value anticlockwise
                front += 1
            elif np.deg2rad(180) > robot[1] > np.deg2rad(
                    100):  # np.deg2rad(260) > robot[1] > np.deg2rad(180): value clockwise, np.deg2rad(180) > robot[1] > np.deg2rad(100): value anticlockwise
                back += 1
        # self.publish_info('#:' + str(len(robots)) + ' front:' + str(front) + ' back:' + str(back))
        if front > 0 and back > 0:
            self.quorum = True
        else:
            self.quorum = False

    def update_is_closure(self):
        """ sets closure to true, if the robots get_surround_direction the object """
        # TODO: mit paper überprüfen ob richtig implementiert
        robots_in_quorum = sum(value == 'True' for value in self.quorum_dict.values())
        min_robots_for_closure = (2 * np.pi * self.get_r_cage()) / (2 * self.r + self.d_min)
        # self.publish_info(
        #     '#quorum=' + str(robots_in_quorum) + '  #min=' + str(min_robots_for_closure) + ' r_cage=' + str(
        #         self.get_r_cage()))
        if robots_in_quorum < min_robots_for_closure:
            self.closure = False
        else:
            self.closure = True

        self.robots_in_quorum = robots_in_quorum

    def update_is_near_object(self):
        # self.publish_info('update_is_near_object')
        # TODO: - self.param_distance_reached muss eventuell abhängig von d_min und d_max sein
        # TODO: sicher gehen das das rechts neben mir kein anderer Roboter ist
        # 0° ist vorne, 90° ist links, 180° ist hinten, 270° ist rechts
        dist = get_distance(self.current_scan, 90)
        self.dist = dist
        # dist = np.linalg.norm(self.current_pose[0:1]-self.current_object_center)
        desired_dist = self.get_r_cage() - 1.0
        robots, robots_center = get_neighbors(self.current_scan, self.get_r_cage() - self.d_min)

        dist_valid = True
        for robot in robots_center:
            if np.pi * 0.5 - 1 < robot[1] < np.pi * 0.5 + 1:
                dist_valid = False

        if not dist_valid:
            self.near_object = self.near_object
        elif self.state == State.APPROACH:
            if dist < desired_dist + 0.5:  # self.get_r_cage() - 0.5 * self.d_min:
                self.near_object = True
            else:
                self.near_object = False
        elif self.state == State.SURROUND:
            if dist < desired_dist + 1.0:  # self.get_r_cage():  # self.param_distance_reached:
                self.near_object = True
            else:
                self.near_object = False

    def update_goal_position(self):
        x_list = []
        y_list = []

        for value in self.goal_dict.values():
            if value != 'False':
                x, y = map(float, value.split(';'))
                x_list.append(x)
                y_list.append(y)

        if len(x_list) == 0:
            if not self.search_goal_timer.is_alive():
                self.search_goal_timer = Timer(self.param_goal_timer_period, self.is_goal_visible,
                                               args=(self.lower_goal_color, self.upper_goal_color))
                self.search_goal_timer.start()
        else:
            if self.search_goal_timer.is_alive():
                self.search_goal_timer.cancel()
                self.search_goal_timer = Timer(self.param_goal_timer_period, self.is_goal_visible,
                                               args=(self.lower_goal_color, self.upper_goal_color))

            # TODO: Wieder einkommentieren, wenn die Abschätzung besser ist
            # self.goal_position[0] = float(np.mean(x_list))
            # self.goal_position[1] = float(np.mean(y_list))

    def calculate_goal_position(self):
        scan_msg = self.current_scan
        image = self.current_image
        lower_color = self.lower_goal_color
        upper_color = self.upper_goal_color

        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        obj_mask = cv.inRange(hsv, lower_color, upper_color)
        x, y = get_centroid(obj_mask)

        # normiert Wert: 1 (ganz links) und -1 (ganz rechts)
        img_width = image.shape[1]
        turn_intensity_and_direction = (2 * x / img_width - 1) * -1

        # Camera module v 2.1: Horizontal field of view = 62.2 deg & Vertical field of view = 48.8 deg
        centroid_degree = turn_intensity_and_direction * 31.1

        distance = get_distance(scan_msg, centroid_degree)
        if abs(distance) > scan_msg.range_max:
            distance = scan_msg.range_max * np.sign(distance)
        x = self.current_pose[0] + distance * np.cos(centroid_degree)
        y = self.current_pose[1] + distance * np.sin(centroid_degree)

        goal_position = str(x) + ";" + str(y)
        return goal_position

    def is_object_visible(self, lower_color, upper_color):
        self.search_object_timer.cancel()
        self.search_object_timer = Timer(self.param_object_timer_period, self.is_object_visible,
                                         args=(self.lower_object_color, self.upper_object_color))

        in_image = self.object_in_image

        if (not self.turn_timer.is_alive()) and (self.state == State.SEARCH):
            # self.publish_info('!!!Starte Überprüfung auf OBJEKT!!!')
            self.turn_once()
            for i in range(0, 1000):
                time.sleep(0.1)
                in_image = is_color_in_image(self.current_image, lower_color, upper_color)
                # self.info.data = 'Objekt Bild ' + str(i)
                # self.information_publisher.publish(self.info)

                if not self.turn_timer.is_alive():
                    # self.publish_info('!!!KEIN OBJEKT in Sicht!!!')
                    in_image = False
                    break

                elif in_image:
                    # self.publish_info('!!!OBJEKT in Sicht!!!')

                    in_image = True
                    self.turn_timer.cancel()
                    self.turn_timer = Timer(self.param_turn_timer_period, self.stop)
                    # self.publish_info('Der turn_timer wurde gecancelt')
                    self.stop()
                    break

        self.object_in_image = in_image

        self.search_object_timer.start()

    def is_goal_visible(self, lower_color, upper_color):
        in_image = False
        if not self.turn_timer.is_alive():
            # self.publish_info('!!!Starte Überprüfung auf ZIEL!!')
            self.drive_backwards()
            self.turn_once()
            for i in range(0, 1000):
                # self.publish_info('Ziel Bild ' + str(i))
                in_image = is_color_in_image(self.current_image, lower_color, upper_color)
                time.sleep(0.1)
                if not self.turn_timer.is_alive() or in_image:
                    break

            # if in_image:
            #     self.publish_info('!!!ZIEL in Sicht!!!')
            # else:
            #     self.publish_info('!!!KEIN ZIEL in Sicht!!!')

        return in_image

    def update_object_infos(self, scan_msg, image, lower_color, upper_color):
        is_in_image = is_color_in_image(image, lower_color, upper_color)
        obj_distance = np.inf
        obj_direction = 0.0

        if is_in_image:
            hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
            obj_mask = cv.inRange(hsv, lower_color, upper_color)
            x, y = get_centroid(obj_mask)

            cv.imwrite('Original.jpeg', image)
            cv.imwrite('Maske.jpeg', obj_mask)

            # normiert Wert: 1 (ganz links) und -1 (ganz rechts)
            img_width = image.shape[1]
            turn_intensity_and_direction = (2 * x / img_width - 1) * -1

            # Camera module v 2.1: Horizontal field of view = 62.2 deg & Vertical field of view = 48.8 deg
            centroid_degree = turn_intensity_and_direction * 31.1

            obj_distance = get_distance(scan_msg, centroid_degree)
            obj_direction = turn_intensity_and_direction * self.param_max_rotational_velocity

            # self.publish_info('relative Centroid: x=' + str(x / img_width) + '  y=' + str(
            #     y / img_width) + '  centroid_degree= ' + str(centroid_degree) + '  Distanz=' + str(obj_distance))

        self.object_in_image = is_in_image
        self.object_direction = obj_direction
        self.object_distance = obj_distance

    # BERECHNUNGEN #
    def update_Q(self):
        # gibt Liste zurück mit der eigenen Position an erster Stelle und danach die Positionen aller Nachbarn (alle Roboter die sich in der Range R_0 um den Roboter befinden)
        robots, robots_center = get_neighbors(self.current_scan, self.R)
        q = [[self.current_pose[0], self.current_pose[1]]]
        for robot in robots_center:
            r = robot[0]
            phi = robot[1] + self.current_pose[2]
            x = r * np.cos(phi) + self.current_pose[0]
            y = r * np.sin(phi) + self.current_pose[1]
            q.append([x, y])

        self.Q = q

    def get_r_cage(self):
        return 0.5 * self.d_max + self.r + self.l + self.e

    def gradient_cross_psi(self, q):
        matrix_3x3 = np.cross(nd.Gradient(self.psi)(q), np.array([1, 1, 1]))
        vector_2d = np.array([matrix_3x3[2][0], matrix_3x3[2][1]])
        return vector_2d

    def gamma(self, q):
        if self.state == State.TRANSPORT:
            # TODO: distance muss abhängig von object durchmesser gesetzt werden
            #  oder über die Zeit des Transports verändert werden wenn sich das Objekt nicht bewegt
            center = get_trajectory(self.current_object_center, self.goal_position, 0.75)
        else:
            center = self.current_object_center

        return s(q[0], q[1], self.get_r_cage(), center)

    def phi(self, q):
        beta_0 = self.R - np.linalg.norm(self.Q)
        gamma_q = self.gamma(q)
        return gamma_q ** 2 / (gamma_q ** 2 + beta_0)

    def psi(self, q):
        return np.array([[0.0, 0.0, self.gamma(q)]]).T

    def t(self, j, k):
        q = self.Q

        multiplier_1 = np.subtract(q[0], q[j]).T
        multiplier_2 = np.multiply(self.gradient_cross_psi(q[j]), -1)
        numerator = np.multiply(multiplier_1, multiplier_2)

        minute = np.linalg.norm(np.subtract(q[0], q[j]), ord=k)
        subtrahend = np.power((2 * self.r), k)
        denominator = np.subtract(minute, subtrahend)

        return np.divide(numerator, denominator)

    def n(self, j, k):
        q = self.Q

        multiplier_1 = np.subtract(q[0], q[j]).T
        multiplier_2 = np.multiply(nd.Gradient(self.phi)(q[j]), -1)
        numerator = np.multiply(multiplier_1, multiplier_2)

        minute = np.linalg.norm(np.subtract(q[0], q[j]), ord=k)
        subtrahend = np.power((2 * self.r), k)
        denominator = np.subtract(minute, subtrahend)

        return np.divide(numerator, denominator)

    def T(self):
        q = self.Q
        result = np.array([0.0, 0.0])
        for j in range(1, len(q)):
            numerator = sigma_plus(self.t(j, 2))
            denominator = np.linalg.norm(np.subtract(q[0], q[j]), ord=2) - ((2 * self.r) ** 2)
            minute = np.divide(numerator, denominator)

            numerator = sigma_minus(self.t(j, 4))
            denominator = np.linalg.norm(np.subtract(q[0], q[j]), ord=4) - ((2 * self.r) ** 4)
            subtrahend = np.divide(numerator, denominator)

            result += np.subtract(minute, subtrahend)

        return result

    def N(self):
        q = self.Q
        result = np.array([0.0, 0.0])
        for j in range(1, len(q)):
            numerator = sigma_plus(self.n(j, 2))
            denominator = np.linalg.norm(np.subtract(q[0], q[j]), ord=2) - ((2 * self.r) ** 2)
            minute = np.divide(numerator, denominator)

            numerator = sigma_minus(self.n(j, 4))
            denominator = np.linalg.norm(np.subtract(q[0], q[j]), ord=4) - ((2 * self.r) ** 4)
            subtrahend = np.divide(numerator, denominator)

            result += np.subtract(minute, subtrahend)

        return result

    # calculating the object shape (concave/convex, height, d_min, d_max, weight)
    def moveAroundObject(self):
        moveAroundObject = Twist()

        if self.start == np.inf:
            self.start = len(self.odometry_list) - 1
        elif self.start == -1:
            return moveAroundObject

        position = self.odometry_list[-1][0]
        start_position = self.odometry_list[self.start][0]
        distance_to_start = np.linalg.norm(np.subtract(position, start_position))

        if len(self.odometry_list) > self.start + 300 and (len(self.odometry_list) > self.start + 2000 or distance_to_start < 0.4):
            self.shape = self.getShape(self.odometry_list[self.start:-1])
            self.publish_info("shape = " + str(self.shape))
            self.start = -1
            moveAroundObject.angular.z = 0.0
            moveAroundObject.linear.x = 0.0
        else:
            moveAroundObject = self.wall_follow()

        return moveAroundObject

    def getShape(self, odometry_list):
        shape = [[], []]
        for element in odometry_list:
            robot_position = element[0]
            robot_orientation = element[1]

            scan = element[2]
            start = int(np.deg2rad(80) / scan.angle_increment)
            end = int(np.deg2rad(100) / scan.angle_increment)
            for i in range(start, end):
                angle = i * scan.angle_increment - robot_orientation
                dist = scan.ranges[i]

                if (not np.isinf(dist)) and (not np.isnan(dist)) and abs(dist) <= scan.range_max:
                    object_border_x = robot_position[0] + dist * np.cos(angle)
                    object_border_y = robot_position[1] + dist * np.sin(angle)

                    shape[0].append(object_border_x)
                    shape[1].append(object_border_y)

            # object_dist = element[2]
            # angle = np.pi * 0.5 - robot_orientation
            #
            # object_border_x = robot_position[0] + object_dist * np.cos(angle)
            # object_border_y = robot_position[1] + object_dist * np.sin(angle)

            # shape[0].append(object_border_x)
            # shape[1].append(object_border_y)

            # shape[0].append(robot_position[0])
            # shape[1].append(robot_position[1])

        # TODO: smooth shape

        return shape

    def wall_follow(self):
        """calculates the vector to follow the left wall"""
        scan_msg = self.current_scan
        left_mean_dist = get_mean_dist(scan_msg, 30, 80)
        front_mean_dist = get_mean_dist(scan_msg, -30, 30)

        desired_dist = 2.0  # desired distance to the wall
        error = left_mean_dist - desired_dist

        linear = self.param_max_translational_velocity
        angular = error * 1.5

        if front_mean_dist < desired_dist + 0.5:
            # turn right:
            angular = -1.5

        linear, angular = self.scale(linear, angular)
        wall_follow_direction = Twist()
        wall_follow_direction.angular.z = angular
        wall_follow_direction.linear.x = linear

        return wall_follow_direction


def main(args=None):
    """Create a node for the caging and handle the setup."""
    setup_node.init_and_spin(args, CagingPattern)


if __name__ == '__main__':
    main()
