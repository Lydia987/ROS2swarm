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

import numpy as np
import time
from threading import Timer
import imutils
import cv2 as cv
from cv_bridge import CvBridge

bridge = CvBridge()


def get_shape_points(odometry_list):
    """ calculates points of the shape from the laser data and robot positions contained in the odometry_list """
    x = []
    y = []
    for j in range(len(odometry_list)):
        robot_position = odometry_list[j][0]
        robot_orientation = odometry_list[j][1]
        scan = odometry_list[j][2]

        start = int(np.deg2rad(83) / scan.angle_increment)
        end = int(np.deg2rad(97) / scan.angle_increment)
        for i in range(start, end):
            angle = i * scan.angle_increment
            dist = scan.ranges[i]

            if (not np.isinf(dist)) and (not np.isnan(dist)) and abs(dist) < scan.range_max:
                x_r = dist * np.cos(angle)
                y_r = dist * np.sin(angle)

                x_g = robot_position[0] + x_r * np.cos(robot_orientation) - y_r * np.sin(robot_orientation)
                y_g = robot_position[1] + x_r * np.cos(robot_orientation) - y_r * np.sin(robot_orientation)

                x.append(x_g)
                y.append(y_g)
    return x, y


def points2image(x, y):
    """ returns an image created from the passed points """
    scale = 100
    frame = 200 / scale

    # Move all points to the positive area
    x = np.add(x, np.min(x) * -1 + frame) * scale
    y = np.add(y, np.min(y) * -1 + frame) * scale

    # Create empty image
    img = np.zeros((int(np.max(y) + frame * scale), int(np.max(x) + frame * scale)), dtype=np.uint8)

    # Draw all points in the image
    for point in zip(*[x, y]):
        cv.circle(img, (int(point[0]), int(point[1])), 1, 255, -1)

    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    return img


def rectify_img(img):
    """ rectifies the image """
    angle = 14.8  # Clockwise rotation angle
    stretch = 0.526
    height, width = img.shape[:2]

    # rotate
    M = cv.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    img = cv.warpAffine(img, M, (width, height))

    # stretch
    new_height = int(height * stretch)

    # scale
    img = cv.resize(img, (width, new_height))

    # rotate back
    M = cv.getRotationMatrix2D((width / 2, new_height / 2), 360 - angle, 1)
    img = cv.warpAffine(img, M, (width, new_height))
    return img


def get_shape_contours(img):
    """ returns a list with the contours contained in the image """
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    for i in range(4):
        # Smooth image
        median = cv.medianBlur(gray, 5)
        blurred = cv.GaussianBlur(median, (5, 5), 0)

        # Close gaps
        kernel = np.ones((5, 5), np.uint8)
        closing = cv.morphologyEx(blurred, cv.MORPH_CLOSE, kernel)
        opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)

        # Apply threshold
        thresh = cv.threshold(opening, 1, 255, cv.THRESH_BINARY)[1]
        gray = thresh

    edges = cv.Canny(gray, 100, 200)

    # Extract contours
    contours = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    return contours


def get_approx_dmin(img, evaluation_path, robot_name):
    """ returns an approximated shape and its smallest diameter """
    contours = get_shape_contours(img)
    contour = None
    w = 0.0
    w_new = 0.0
    h = 0.0
    h_new = 0.0
    rect = None
    rect_new = None
    approx = None

    # Find largest contour
    for contour in contours:
        # Calculate surrounding rectangle
        rect_new = cv.minAreaRect(contour)
        w_new = rect_new[1][0]
        h_new = rect_new[1][1]

        if w_new * h_new > w * h:
            w = w_new
            h = h_new
            rect = rect_new
            # approximate contour
            approx = cv.approxPolyDP(contour, 15, True)

    d_min = min(w, h)

    # only for evaluation purposes
    box = np.intp(cv.boxPoints(rect))
    cv.drawContours(img, [box], 0, (0, 255, 0), 2)
    cv.drawContours(img, [approx], -1, (0, 0, 255), 2)
    cv.imwrite(evaluation_path + "EvaluatedShape_" + robot_name + ".jpeg", img)

    return approx, d_min / 100


def approx2contour(approx):
    """ returns the contour of the approximated shape """
    contour = []
    line_points = []

    for i in range(len(approx) - 1):
        p1 = approx[i][0]
        p2 = approx[i + 1][0]
        # Determination of the slope
        if p2[0] - p1[0] != 0:
            m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        else:
            m = np.sign(p2[1] - p1[1]) * 9999999999999999

        # Determination of the Y-axis intercept
        b = p1[1] - m * p1[0]

        # Determination of the larger and smaller point
        if p1[0] < p2[0]:
            x1 = p1[0]
            x2 = p2[0]
        else:
            x1 = p2[0]
            x2 = p1[0]

        for x in range(int(x1), int(x2) + 1):
            y = m * x + b
            line_points.append([x, y])

        contour.extend(line_points)

    contour = np.array(contour).reshape((-1, 1, 2)).astype(np.float32)
    return contour


def get_dmax(contour, evaluation_path, robot_name):
    """ Returns the maximum diameter of the contour. """
    d_max = 0.0

    # for evaluation purposes only
    max_line = []

    for i in range(len(contour)):
        for j in range(i, len(contour)):
            p1 = contour[i][0]
            p2 = contour[j][0]
            dist = np.linalg.norm(p1 - p2)
            if dist > d_max:
                d_max = dist
                # for evaluation purposes only
                max_line = [(int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))]

    # draws d_max in img, for evaluation purposes only
    img = cv.imread(evaluation_path + "EvaluatedShape_" + robot_name + ".jpeg")
    cv.line(img, max_line[0], max_line[1], (255, 255, 0), 2)
    cv.imwrite(evaluation_path + "EvaluatedShape_" + robot_name + ".jpeg", img)

    return d_max / 100


def get_neighbors(scan_msg, max_range):
    """ returns all robots in the radius max_range """
    # TODO: Create params for threshold and width depending on robot type
    # robots are detected between 0.2m and 3.5m min_range=0, max_range=3.5, threshold=0.35, min_width=0, max_width=15
    robots, robots_center = ScanCalculationFunctions.identify_robots(laser_scan=scan_msg, min_range=0,
                                                                     max_range=max_range, threshold=0.35, min_width=0,
                                                                     max_width=15)
    return robots, robots_center


def get_image_contour(img, lower_color, upper_color):
    """ returns a contour if exactly one contour is detected in the image """
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    grey = cv.inRange(hsv, lower_color, upper_color)
    median = cv.medianBlur(grey, 5)
    blurred = cv.GaussianBlur(median, (5, 5), 0)
    thresh = cv.threshold(blurred, 60, 255, cv.THRESH_BINARY)[1]

    contours = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    if len(contours) == 1:
        return contours[0]
    else:
        return None


def get_center(contour):
    """ Returns the center of the contour in pixel. """
    x = 0
    y = 0
    M = cv.moments(contour)

    if M["m00"] != 0:
        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])

    return x, y


def is_color_in_image(image, lower_color, upper_color):
    """ returns true if the image contains the given color """
    is_in_image = False
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    obj_mask = cv.inRange(hsv, lower_color, upper_color)
    number_of_white_pix = np.sum(obj_mask == 255)
    number_of_pix = obj_mask.shape[0] * obj_mask.shape[1]
    percentage_of_white_pix = number_of_white_pix * (100 / number_of_pix)
    if percentage_of_white_pix > 0.01:
        is_in_image = True
    return is_in_image


def get_distance(scan_msg, degree):
    """ returns the distance in the direction of the given angle """
    rad = np.deg2rad(degree)
    increment = scan_msg.angle_increment
    min_rad = scan_msg.angle_min
    max_rad = scan_msg.angle_max
    if rad < min_rad or rad > max_rad:
        distance = np.inf
    else:
        distance = scan_msg.ranges[int((rad - min_rad) / increment)]
    return distance


def get_mean_dist(scan_msg, min_angle, max_angle):
    """ calculates the mean distance to obstacles between the min_angle and max_angle """
    max_range = scan_msg.range_max
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
        if np.isinf(dist_i) or dist_i > max_range:
            dist_i = max_range

        sum_dist += dist_i

    mean_dist = sum_dist / (end - start)
    return mean_dist


class DynamicChangeTransportPattern(MovementPattern):
    """
    Pattern to transport an object to the target.
    """

    def __init__(self):
        """Initialize the dynamic change transport pattern node."""
        super().__init__('dynamic_change_transport_pattern')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('turn_timer_period', None),
                ('object_search_timer_period', None),
                ('object_distance_reached', None),
                ('max_translational_velocity', None),
                ('max_rotational_velocity', None),
                ('robot_radius', None),
                ('robot_length', None),
                ('max_transport_time', None),
                ('max_detectable_height', None),
                ('min_detectable_height', None),
                ('pixel_size', None),
                ('object_name', None),
                ('goal_name', None),
                ('evaluation_path', None)
            ])

        # PARAMS #
        self.object_name = self.get_parameter("object_name").get_parameter_value().string_value
        self.goal_name = self.get_parameter("goal_name").get_parameter_value().string_value
        self.param_max_translational_velocity = self.get_parameter(
            "max_translational_velocity").get_parameter_value().double_value
        self.param_max_rotational_velocity = self.get_parameter(
            "max_rotational_velocity").get_parameter_value().double_value
        self.param_turn_timer_period = self.get_parameter("turn_timer_period").get_parameter_value().double_value
        self.param_object_timer_period = self.get_parameter(
            "object_search_timer_period").get_parameter_value().double_value
        self.r = self.get_parameter("robot_radius").get_parameter_value().double_value
        self.l = self.get_parameter("robot_length").get_parameter_value().double_value
        self.max_transport_time = self.get_parameter("max_transport_time").get_parameter_value().double_value
        self.max_detectable_height = self.get_parameter("max_detectable_height").get_parameter_value().double_value
        self.min_detectable_height = self.get_parameter("min_detectable_height").get_parameter_value().double_value
        self.pixel_size = self.get_parameter("pixel_size").get_parameter_value().double_value
        self.evaluation_path = self.get_parameter("evaluation_path").get_parameter_value().string_value

        # SUBPATTERN #
        self.random_walk_latest = Twist()
        self.random_walk_subpattern = self.create_subscription(Twist,
                                                               self.get_namespace() + '/drive_command_random_walk_pattern',
                                                               self.command_callback_random_walk, 10)
        self.pushing_latest = Twist()
        self.pushing_subpattern = self.create_subscription(Twist,
                                                           self.get_namespace() + '/drive_command_pushing_pattern',
                                                           self.command_callback_pushing, 10)
        self.caging_latest = Twist()
        self.caging_subpattern = self.create_subscription(Twist,
                                                          self.get_namespace() + '/drive_command_caging_pattern',
                                                          self.command_callback_caging, 10)
        # SUBSCRIBER #
        self.odom_subscription = self.create_subscription(Odometry, self.get_namespace() + '/odom', self.odom_callback,
                                                          qos_profile=qos_profile_sensor_data)
        self.scan_subscription = self.create_subscription(
            LaserScan, self.get_namespace() + '/scan', self.swarm_command_controlled(self.scan_callback),
            qos_profile=qos_profile_sensor_data)
        self.camera_subscription = self.create_subscription(
            Image, self.get_namespace() + '/camera/image_raw', self.swarm_command_controlled(self.camera_callback),
            qos_profile=qos_profile_sensor_data)

        # PUBLISHER #
        self.information_publisher = self.create_publisher(StringMessage, self.get_namespace() + '/information', 10)
        self.protection_publisher = self.create_publisher(StringMessage, self.get_namespace() + '/hardware_protection_layer', 10)

        # GENERAL #
        self.direction = Twist()
        self.state = State.INIT
        self.pushing = False
        self.caging = False
        self.current_scan = None
        self.current_image = None
        self.current_pose = [0.0, 0.0, 0.0]  # [x, y, orientation] in [m, m, rad]
        self.odometry_list = []
        self.start_index_survey = np.inf  # index of the list at which the survey of the object was started
        self.max_transport_time_reached = False
        self.transport_start_time = None

        # OBJECT #
        self.lower_object_color = np.array([50, 50, 50])  # in HSV
        self.upper_object_color = np.array([70, 255, 255])  # in HSV
        self.near_object = False
        self.near_object_counter = 0
        self.object_in_image = False
        self.object_direction = None
        self.object_height = 0.0
        self.density = 1.0  # kg * m³

        self.info = StringMessage()
        self.info.data = 'Starting Dynamic Change Transport Pattern'
        self.information_publisher.publish(self.info)

        self.protection = StringMessage()
        self.protection.data = 'True'
        self.protection_publisher.publish(self.protection)

        # TIMER #
        self.search_object_timer = Timer(self.param_object_timer_period, self.is_object_visible)
        self.turn_timer = Timer(self.param_turn_timer_period, self.stop)

        # TEST AND EVALUATION #
        self.model_states_subscription = self.create_subscription(ModelStates, '/gazebo/model_states',
                                                                  self.model_states_callback,
                                                                  qos_profile=qos_profile_sensor_data)
        # 0 = time, 1 = velocity, 2 = state, 3 = pose, 4 = object_center, 5 = goal_position, 6 = shape_parameter]
        self.state_list = [[], [[], []], [], [[], [], []], [[], [], []], [[], []], []]
        self.publish_state_counter = 0
        self.current_object_center = [[], [], []]
        self.goal_position = [[], []]
        self.current_pose = [[], [], []]
        self.robot_name = str(self.get_namespace())[-1]

    # PUBLISHER #
    def save_evaluation_data(self):
        """ saves all evaluation data in a txt file """
        self.state_list[0].append([time.time()])
        self.state_list[1][0].append(self.direction.linear.x)
        self.state_list[1][1].append(self.direction.angular.z)
        self.state_list[2].append([str(self.state)])
        self.state_list[3][0].append(self.current_pose[0])
        self.state_list[3][1].append(self.current_pose[1])
        self.state_list[3][2].append(np.rad2deg(self.current_pose[2]))
        self.state_list[4][0].append(self.current_object_center[0])
        self.state_list[4][1].append(self.current_object_center[1])
        self.state_list[4][2].append(np.rad2deg(self.current_object_center[2]))
        self.state_list[5][0].append(self.goal_position[0])
        self.state_list[5][1].append(self.goal_position[1])
        with open('dynamic_change_state_list_' + str(self.get_namespace())[-1] + '.txt', 'w') as doc:
            doc.write(str(self.state_list))

        # self.publish_state_counter -= 1
        # if self.publish_state_counter <= 0:
        #     self.publish_state_counter = 10
        #     self.info.data = '_____' + str(self.state) + '_____ height=' + str(self.object_height) + ' near_object=' + str(self.near_object) + ' object_in_image=' + str(self.object_in_image)
        #     self.information_publisher.publish(self.info)

    def publish_info(self, msg):
        self.info.data = str(msg)
        self.information_publisher.publish(self.info)

    # CALLBACKS #
    def model_states_callback(self, model_states):
        """Call back if a new model_states msg is available."""
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

                z = model_states.pose[index].orientation.z
                w = model_states.pose[index].orientation.w

                if np.sign(z) < 0:  # clockwise
                    self.current_object_center[2] = 2 * np.pi - 2 * np.arccos(w)
                else:  # anticlockwise
                    self.current_object_center[2] = 2 * np.arccos(w)

            elif name == self.goal_name:
                self.goal_position[0] = model_states.pose[index].position.x
                self.goal_position[1] = model_states.pose[index].position.y
            index += 1

    def odom_callback(self, odom_msg):
        """Call back if a new odom msg is available."""
        if self.current_scan is not None and self.state == State.SURVEY_OBJECT:
            currentPosition = [odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y]
            z = odom_msg.pose.pose.orientation.z
            w = odom_msg.pose.pose.orientation.w
            if np.sign(z) < 0:  # clockwise
                currentOrientation = 2 * np.pi - 2 * np.arccos(w)
            else:  # anticlockwise
                currentOrientation = 2 * np.arccos(w)
            self.odometry_list.append([currentPosition, currentOrientation, self.current_scan])

    def command_callback_random_walk(self, incoming_msg: Twist):
        """Assign the message to variable"""
        self.random_walk_latest = incoming_msg

    def command_callback_caging(self, incoming_msg: Twist):
        """Assign the message to variable"""
        self.caging_latest = incoming_msg

    def command_callback_pushing(self, incoming_msg: Twist):
        """Assign the message to variable"""
        self.pushing_latest = incoming_msg

    def scan_callback(self, incoming_msg: LaserScan):
        """Call back if a new scan msg is available."""
        self.current_scan = incoming_msg
        self.dynamic_change()

    def camera_callback(self, raw_image_msg: Image):
        """Call back if a new scan msg is available."""
        self.current_image = bridge.imgmsg_to_cv2(raw_image_msg, desired_encoding='passthrough')

    # STATUS UPDATE AND EXECUTION #
    def dynamic_change(self):
        self.save_evaluation_data()

        if self.max_transport_time_reached:
            self.state = State.STOP
        elif self.is_busy():
            return

        self.update_flags()
        self.update_state()
        self.execute_state()

    def update_flags(self):
        if self.state != State.INIT:
            self.update_is_max_transport_time_reached()

            if self.state == State.SEARCH or self.state == State.APPROACH:
                self.update_is_object_in_image()

            if self.state == State.APPROACH or self.state == State.SURVEY_OBJECT:
                self.update_is_near_object()

    def update_state(self):
        if self.state != State.INIT:
            if self.state == State.SEARCH:
                if self.object_in_image:
                    self.state = State.APPROACH

            elif self.state == State.APPROACH:
                if not self.object_in_image:
                    self.state = State.SEARCH
                elif self.near_object:
                    self.state = State.SURVEY_OBJECT

            elif self.state == State.SURVEY_OBJECT:
                if not self.near_object:
                    self.state = State.APPROACH
                elif self.caging:
                    self.state = State.CAGING
                elif self.pushing:
                    self.state = State.PUSHING

            elif self.state == State.PUSHING or self.state == State.CAGING:
                if self.max_transport_time_reached:
                    self.state = State.STOP

    def execute_state(self):
        """ Executes behavior depending on status. """
        if self.state == State.INIT:
            fully_initialized = not (self.current_image is None or self.current_scan is None)
            if fully_initialized:
                self.search_object_timer.start()
                self.direction = self.random_walk_latest
                self.state = State.SEARCH
                self.protection.data = 'True'

        elif self.state == State.SEARCH:
            self.direction = self.random_walk_latest
            self.protection.data = 'True'

        elif self.state == State.APPROACH:
            self.update_object_height()
            self.direction = self.approach()
            self.protection.data = 'True'

        elif self.state == State.SURVEY_OBJECT:
            self.direction = self.survey_object()
            self.protection.data = 'True'

        elif self.state == State.CAGING:
            self.direction = self.caging_latest
            self.protection.data = 'False'

        elif self.state == State.PUSHING:
            self.direction = self.pushing_latest
            self.protection.data = 'False'

        elif self.state == State.STOP:
            self.direction = Twist()
            self.protection.data = 'False'

        self.protection_publisher.publish(self.protection)
        self.command_publisher.publish(self.direction)

    # MOVEMENT PATTERNS #
    def approach(self):
        """ drives towards object center """
        image = self.current_image
        img_width = image.shape[1]
        contour = get_image_contour(image, self.lower_object_color, self.upper_object_color)
        if contour is not None:
            x, y = get_center(contour)
            # normiert Wert: 1 (ganz links) und -1 (ganz rechts)
            turn_intensity_and_direction = (2 * x / img_width - 1) * -1
            # draw the contour and center of the shape on the image, for evaluation only
            cv.drawContours(image, [contour], -1, (255, 0, 0), 2)
            cv.circle(image, (x, y), 7, (255, 0, 0), -1)
            cv.imwrite(self.evaluation_path + 'ApproachCenter_' + self.robot_name + '.jpeg', image)
        else:
            turn_intensity_and_direction = 0.0

        approach = Twist()
        approach.angular.z = turn_intensity_and_direction * self.param_max_rotational_velocity
        approach.linear.x = self.param_max_translational_velocity

        return approach

    def survey_object(self):
        """ Travels around the object at least once and measures it in the process """
        survey_object = Twist()
        if len(self.odometry_list) > 0:
            if self.start_index_survey == np.inf:
                self.start_index_survey = len(self.odometry_list) - 1
            elif self.start_index_survey == -1:
                return survey_object

            position = self.odometry_list[-1][0]
            start_position = self.odometry_list[self.start_index_survey][0]
            distance_to_start = np.linalg.norm(np.subtract(position, start_position))
            # Possibly check whether it has already been near the start position twice.

            if self.start_index_survey > -1 and len(self.odometry_list) > self.start_index_survey + 300 and (
                    len(self.odometry_list) > self.start_index_survey + 1500):  # or distance_to_start < 0.01):
                self.decide_transport_strategy(self.odometry_list[self.start_index_survey + 200:-1])
                self.start_index_survey = -1
            else:
                survey_object = self.wall_follow()

        return survey_object

    def wall_follow(self):
        """calculates the vector to follow the left wall"""

        scan_msg = self.current_scan
        left_mean_dist = get_mean_dist(scan_msg, 30, 80)
        front_mean_dist = get_mean_dist(scan_msg, -30, 30)
        desired_dist = 1.9  # desired distance to the wall
        error = left_mean_dist - desired_dist

        linear = self.param_max_translational_velocity
        angular = error * 1.5

        if front_mean_dist < desired_dist + 0.65:
            # turn right:
            angular = -1.0

        robots, robots_center = get_neighbors(self.current_scan, 1.6)
        min_dist = np.inf
        for robot in robots_center:
            if np.deg2rad(340) < robot[1] < np.deg2rad(20) and robot[0] < min_dist:
                min_dist = robot[0]
            if min_dist <= 1.6:
                linear = linear * min_dist * 0.5

        linear, angular = self.scale_velocity(linear, angular)
        wall_follow_direction = Twist()
        wall_follow_direction.angular.z = angular
        wall_follow_direction.linear.x = linear

        return wall_follow_direction

    def stop(self):
        """ Stops the robot. """
        stop = Twist()
        self.command_publisher.publish(stop)

    def turn_once(self):
        """ Turns once. """
        self.turn_timer = Timer(self.param_turn_timer_period, self.stop)
        turn = Twist()
        turn.angular.z = self.param_max_rotational_velocity

        self.command_publisher.publish(turn)
        self.turn_timer.start()

    def is_busy(self):
        if self.turn_timer.is_alive() and self.state != State.INIT:
            return True
        else:
            return False

    def is_object_visible(self):
        """ Rotates a maximum of once and stops as soon as the object is included in the image and stets object_in_image to true if the object was in the image. """
        self.search_object_timer.cancel()
        self.search_object_timer = Timer(self.param_object_timer_period, self.is_object_visible)

        if (not self.turn_timer.is_alive()) and (self.state == State.SEARCH):
            self.turn_once()
            for i in range(0, 1000):
                time.sleep(0.1)
                self.update_is_object_in_image()

                if not self.turn_timer.is_alive():
                    break

                elif self.object_in_image:
                    self.turn_timer.cancel()
                    self.turn_timer = Timer(self.param_turn_timer_period, self.stop)
                    self.stop()
                    break

        self.search_object_timer.start()

    def update_object_height(self):
        """ Sets object_height to the calculated object height. """

        if self.object_height >= self.min_detectable_height:
            return

        scan = self.current_scan
        img = self.current_image
        dist = get_distance(scan, 0)

        if dist >= scan.range_max:
            return

        # Checking if there is a robot in front of you
        robots, robots_center = get_neighbors(scan, scan.range_max)
        for robot in robots_center:
            if np.deg2rad(355) < robot[1] < np.deg2rad(5):
                # self.publish_info("robot front: " + str(np.rad2deg(robot[1])))
                return

        contour = get_image_contour(img, self.lower_object_color, self.upper_object_color)
        if contour is None:
            # self.publish_info("contour is None")
            return

        # for evaluation only
        cv.drawContours(img, [contour], -1, (255, 0, 0), 2)
        cv.imwrite(self.evaluation_path + "height_dist" + str(dist) + "_robot" + self.robot_name + ".jpeg", img)

        # get the height of the contour at the center of the picture
        min_y = img.shape[0]
        max_y = 0
        x_center = int(img.shape[1] / 2)
        for y in range(img.shape[0]):
            b, g, r = img[y, x_center]
            if b > 150 and g < 50 and r < 50:
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y

        height_px = max_y - min_y
        height = height_px * self.pixel_size

        if height > self.max_detectable_height:
            height = np.inf
        elif height < self.min_detectable_height:
            height = 0.0

        self.object_height = height

    def update_is_near_object(self):
        old_near_object = self.near_object
        self.near_object = True
        dist_valid = True

        robots, robots_center = get_neighbors(self.current_scan, 1.5)

        if self.state == State.APPROACH:
            dist = get_mean_dist(self.current_scan, -10, 10)
            if dist > 2.0:
                self.near_object = False
            for robot in robots_center:
                if np.deg2rad(350) < robot[1] < np.deg2rad(10):
                    dist_valid = False

        if self.state == State.SURVEY_OBJECT:
            dist = get_distance(self.current_scan, 90)
            for robot in robots_center:
                if np.deg2rad(75) < robot[1] < np.deg2rad(105):
                    dist_valid = False
            if dist > 2.8 or not dist_valid:
                self.near_object_counter += 1
                if self.near_object_counter > 8:
                    self.near_object_counter = 0
                    self.near_object = False
                    self.start_index_survey = np.inf

        if not dist_valid:
            self.near_object = old_near_object

    def update_is_object_in_image(self):
        image = self.current_image
        lower_color = self.lower_object_color
        upper_color = self.upper_object_color
        self.object_in_image = is_color_in_image(image, lower_color, upper_color)

    def update_is_max_transport_time_reached(self):
        if self.transport_start_time is None:
            self.transport_start_time = time.time()

        if time.time() - self.transport_start_time > self.max_transport_time:
            self.max_transport_time_reached = True

    def decide_transport_strategy(self, odometry_list):
        """ Calculates required parameters for the decision and then decides whether to apply pushing or caging by setting pushing or caging to true. """
        # calculate required parameters for the decision
        x, y = get_shape_points(odometry_list)
        img = points2image(x, y)
        cv.imwrite(self.evaluation_path + "OriginalShape_" + self.robot_name + ".jpeg", img)  # evaluation
        img = rectify_img(img)
        cv.imwrite(self.evaluation_path + "RectifiedShape_" + self.robot_name + ".jpeg", img)  # evaluation
        approx, d_min = get_approx_dmin(img, self.evaluation_path, self.robot_name)  # handover of the evaluation_path and the robot_name for evaluation only
        d_max = get_dmax(approx2contour(approx), self.evaluation_path, self.robot_name)  # handover of the evaluation_path and the robot_name for evaluation only
        convex = cv.isContourConvex(approx)
        area = cv.contourArea(approx)
        if self.object_height < self.min_detectable_height:
            weight = area * self.min_detectable_height * self.density  # max
        elif self.object_height > self.max_detectable_height:
            weight = area * self.max_detectable_height * self.density  # min
        else:
            weight = area * self.object_height * self.density
        e = d_max * 0.076
        r_cage = 0.5 * d_max + self.r + self.l + e
        min_robots_caging = (2 * np.pi * r_cage) / (2 * self.r + d_min)
        number_of_available_robots = 5.0  # TODO: Estimate how many robots are available

        pushing = True
        caging = True

        # I came up with the 10Kg and 2kg by testing 5 Turtlebots transporting an object with different transport strategies and weights.
        # TODO: the parameters still have to be made depending on the number of robots and the robot type
        if not convex or weight > 10:
            pushing = False
        if min_robots_caging > number_of_available_robots or weight > 2:
            caging = False
        if caging and pushing:
            pushing = False
        if not pushing and not caging:
            pushing = True

        self.pushing = pushing
        self.caging = caging

        # for evaluation only
        height = self.object_height
        if height == np.inf:
            height = 100.0
        self.state_list[6].append([d_min, d_max, height, area, weight, e, r_cage, convex])

        with open(self.evaluation_path + 'dynamic_change_state_list_' + self.robot_name + '.txt', 'w') as doc:
            doc.write(str(self.state_list))

    def scale_velocity(self, translational_velocity, rotational_velocity):
        """ scales the speed so that the maximum values are not exceeded. """
        if abs(translational_velocity) > self.param_max_translational_velocity:
            scale = self.param_max_translational_velocity / abs(translational_velocity)
            translational_velocity = translational_velocity * scale
            rotational_velocity = rotational_velocity * scale
        if abs(rotational_velocity) > self.param_max_rotational_velocity:
            scale = self.param_max_rotational_velocity / abs(rotational_velocity)
            translational_velocity = translational_velocity * scale
            rotational_velocity = rotational_velocity * scale
        return translational_velocity, rotational_velocity


def main(args=None):
    """Create a node for the caging and handle the setup."""
    setup_node.init_and_spin(args, DynamicChangeTransportPattern)


if __name__ == '__main__':
    main()
