from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image
from communication_interfaces.msg import StringMessage
from rclpy.qos import qos_profile_sensor_data
from ros2swarm.utils import setup_node
from ros2swarm.utils.state import State
from ros2swarm.movement_pattern.movement_pattern import MovementPattern
from ros2swarm.utils.scan_calculation_functions import ScanCalculationFunctions
from cv_bridge import CvBridge
from threading import Timer
import random
import numpy as np
import time
import imutils
import cv2 as cv

bridge = CvBridge()


def get_neighbors(scan_msg, max_range):
    # TODO: Params für den threshold und width abhängig von roboter typ anlegen
    # center: erster wert entfernung in meter , zweiter wert rad (vorne 0, links rum steigend bis 2pi)
    # roboter werden zwischen 0.2m und 3.5m erkannt min_range=0, max_range=3.5, threshold=0.35, min_width=0, max_width=15
    robots, robots_center = ScanCalculationFunctions.identify_robots(laser_scan=scan_msg, min_range=0,
                                                                     max_range=max_range, threshold=0.35, min_width=0,
                                                                     max_width=15)
    return robots, robots_center


def get_image_contour(img, lower_color, upper_color):
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
    x = 0
    y = 0
    M = cv.moments(contour)

    if M["m00"] != 0:
        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])

    return x, y


def get_mean_dist(scan_msg, min_angle, max_angle):
    """ calculates the mean distance to obstacles between the min_angle and max_angle """
    """ calculates the mean distance to obstacles between the min_angle and max_angle """
    R = scan_msg.range_max
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


def has_neighbors(scan_msg):
    # center: erster wert entfernung in meter, zweiter wert rad (vorne 0, links rum steigend bis 2pi)
    # roboter werden zwischen 0,2 m und 3 m erkannt min_range=0, max_range=3, threshold=0.35, min_width=0, max_width=15
    robots, robots_center = ScanCalculationFunctions.identify_robots(laser_scan=scan_msg, min_range=0, max_range=1.5,
                                                                     threshold=0.35, min_width=0, max_width=15)
    left = 0
    right = 0

    for robot in robots_center:
        if np.deg2rad(100) > robot[1] > np.deg2rad(5):
            left += 1
        elif np.deg2rad(355) > robot[1] > np.deg2rad(260):
            right += 1

    return left, right


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


class PushingPattern(MovementPattern):
    """
    Pattern to search and drive to an object with a defined color.
    """

    def __init__(self):
        """Initialize the pushing pattern node."""
        super().__init__('pushing_pattern')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('turn_timer_period', None),
                ('goal_search_timer_period', None),
                ('object_search_timer_period', None),
                ('wall_follow_timer_period', None),
                ('max_transport_time', None),
                ('max_translational_velocity', None),
                ('max_rotational_velocity', None),
                ('object_name', None),
                ('goal_name', None)
            ])

        # PARAMS #
        self.object_name = self.get_parameter("object_name").get_parameter_value().string_value
        self.goal_name = self.get_parameter("goal_name").get_parameter_value().string_value
        self.param_max_translational_velocity = self.get_parameter(
            "max_translational_velocity").get_parameter_value().double_value
        self.param_max_rotational_velocity = self.get_parameter(
            "max_rotational_velocity").get_parameter_value().double_value
        self.param_turn_timer_period = self.get_parameter(
            "turn_timer_period").get_parameter_value().double_value
        self.param_goal_timer_period = self.get_parameter(
            "goal_search_timer_period").get_parameter_value().double_value
        self.param_object_timer_period = self.get_parameter(
            "object_search_timer_period").get_parameter_value().double_value
        self.param_move_around_object_timer_period = self.get_parameter(
            "wall_follow_timer_period").get_parameter_value().double_value
        self.max_transport_time = self.get_parameter("max_transport_time").get_parameter_value().double_value

        # SUBSCRIBER #
        self.scan_subscription = self.create_subscription(LaserScan, self.get_namespace() + '/scan',
                                                          self.swarm_command_controlled(self.scan_callback),
                                                          qos_profile=qos_profile_sensor_data)
        self.camera_subscription = self.create_subscription(Image, self.get_namespace() + '/camera/image_raw',
                                                            self.swarm_command_controlled(self.camera_callback),
                                                            qos_profile=qos_profile_sensor_data)

        # PUBLISHER #
        self.information_publisher = self.create_publisher(StringMessage, self.get_namespace() + '/information', 10)
        self.protection_publisher = self.create_publisher(StringMessage,
                                                          self.get_namespace() + '/hardware_protection_layer', 10)

        # GENERAL #
        self.state = State.INIT
        self.current_scan = None
        self.current_image = None
        self.direction = Twist()
        self.max_transport_time_reached = False
        self.transport_start_time = None
        self.random_walk_latest = Twist()

        # OBJEKT #
        self.lower_object_color = np.array([50, 50, 50])
        self.upper_object_color = np.array([70, 255, 255])
        self.object_in_image = False
        self.near_object = False
        self.object_in_center = False

        # GOAL #
        self.lower_goal_color = np.array([110, 100, 100])
        self.upper_goal_color = np.array([130, 255, 255])
        self.goal_in_image = True
        self.goal_is_occluded = False

        self.info = StringMessage()
        self.info.data = 'Starting pushing pattern'
        self.information_publisher.publish(self.info)

        self.protection = StringMessage()
        self.protection.data = 'True'
        self.protection_publisher.publish(self.protection)

        # TIMER #
        self.search_goal_timer = Timer(self.param_goal_timer_period, self.is_goal_occluded)
        self.last_goal_check = time.time() - 7.0
        self.search_object_timer = Timer(self.param_object_timer_period, self.is_object_visible)
        self.turn_timer = Timer(self.param_turn_timer_period, self.stop)
        self.move_around_object_start_time = None
        # COUNTER #
        self.state_switch_counter = 0

        # TEST und AUSWERTUNG #
        self.model_states_subscription = self.create_subscription(ModelStates, '/gazebo/model_states',
                                                                  self.model_states_callback,
                                                                  qos_profile=qos_profile_sensor_data)
        # 0 = time, 1 = velocity, 2 = state, 3 = pose, 4 = object_center, 5 = goal_position]
        self.state_list = [[], [[], []], [], [[], [], []], [[], [], []], [[], []]]
        self.publish_state_counter = 0
        self.current_object_center = [[], [], []]
        self.goal_position = [[], []]
        self.current_pose = [[], [], []]

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
        self.state_list[4][2].append(np.rad2deg(self.current_object_center[2]))
        self.state_list[5][0].append(self.goal_position[0])
        self.state_list[5][1].append(self.goal_position[1])
        with open('pushing_state_list_' + str(self.get_namespace())[-1] + '.txt', 'w') as doc:
            doc.write(str(self.state_list))

        self.publish_state_counter -= 1
        if self.publish_state_counter <= 0:
            self.publish_state_counter = 10
            self.info.data = '__' + str(self.state) + '__ goal_is_occluded=' + str(
                self.goal_is_occluded) + ' object_in_image=' + str(self.object_in_image)
            self.information_publisher.publish(self.info)

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

    def publish_info(self, msg):
        self.info.data = str(msg)
        self.information_publisher.publish(self.info)

    # CALLBACKS #
    def scan_callback(self, incoming_msg: LaserScan):
        """Call back if a new scan msg is available."""
        self.current_scan = incoming_msg
        self.random_walk_latest.linear.x = self.param_max_translational_velocity
        self.random_walk_latest.angular.z = random.randint(-1, 2) * random.randint(1,
                                                                                   11) * 0.1 * self.param_max_rotational_velocity

        self.pushing()

    def camera_callback(self, raw_image_msg: Image):
        """Call back if a new image msg is available."""
        self.current_image = bridge.imgmsg_to_cv2(raw_image_msg, desired_encoding='passthrough')

    # STATUS AKTUALISIERUNG UND AUSFÜHRUNG #
    def pushing(self):
        self.publish_state()

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
            self.update_is_object_in_image()

            if self.state != State.SEARCH or self.state != State.MOVE_AROUND_OBJECT:
                self.update_is_near_object()

    def update_state(self):
        old_state = self.state

        if self.state != State.INIT:
            if self.state == State.SEARCH:
                if self.object_in_image:
                    self.state = State.APPROACH

            elif self.state == State.APPROACH:
                if not self.object_in_image:
                    self.state = State.SEARCH
                elif self.near_object and self.object_in_center:
                    self.state = State.CHECK_FOR_GOAL

            elif self.state == State.CHECK_FOR_GOAL:
                # if not self.object_in_image:
                #     self.state = State.SEARCH
                if self.goal_is_occluded:
                    self.state = State.PUSH
                else:
                    self.state = State.MOVE_AROUND_OBJECT
                    self.move_around_object_start_time = time.time()

            elif self.state == State.MOVE_AROUND_OBJECT:
                if time.time() - self.move_around_object_start_time >= self.param_move_around_object_timer_period:
                    self.state = State.SEARCH

            elif self.state == State.PUSH:
                # if not self.object_in_image:
                #     self.state = State.SEARCH
                if get_mean_dist(self.current_scan, -5, 5) > 1.5:
                    self.state = State.SEARCH
                elif not self.goal_is_occluded:
                    self.state = State.MOVE_AROUND_OBJECT
                    self.move_around_object_start_time = time.time()

        if self.state == State.SEARCH and old_state == State.APPROACH and self.state_switch_counter < 15:
            self.state = old_state
            self.state_switch_counter += 1
        else:
            self.state_switch_counter = 0

    def execute_state(self):
        # Führt Verhalten abhängig von Status aus
        if self.state == State.INIT:
            fully_initialized = not (self.current_image is None or self.current_scan is None)
            if fully_initialized:
                self.search_object_timer.start()
                self.search_goal_timer.start()
                self.direction = self.random_walk_latest
                self.state = State.SEARCH

        elif self.state == State.SEARCH:
            self.direction = self.avoid(self.random_walk_latest)
            self.protection.data = 'True'

        elif self.state == State.APPROACH:
            self.direction = self.approach()
            self.protection.data = 'True'

        elif self.state == State.PUSH:
            self.direction = self.push()
            self.protection.data = 'False'

        elif self.state == State.MOVE_AROUND_OBJECT:
            self.direction = self.wall_follow()
            self.protection.data = 'True'

        elif self.state == State.CHECK_FOR_GOAL:
            self.search_goal_timer.cancel()
            self.search_goal_timer = Timer(0.0, self.is_goal_occluded)
            self.search_goal_timer.start()
            return
            # self.direction = Twist()
            # self.protection.data = 'False'

        elif self.state == State.STOP:
            self.direction = Twist()
            self.protection.data = 'False'

        self.protection_publisher.publish(self.protection)
        self.command_publisher.publish(self.direction)
        self.publish_info("published command: " + str(self.state))

    # BEWEGUNGSMUSTER #
    def approach(self):
        img = self.current_image
        img_width = img.shape[1]
        contour = get_image_contour(img, self.lower_object_color, self.upper_object_color)
        self.object_in_center = False
        if contour is not None:
            x, y = get_center(contour)
            # normiert Wert: 1 (ganz links) und -1 (ganz rechts)
            turn_intensity_and_direction = (2 * x / img_width - 1) * -1
            if abs(turn_intensity_and_direction) < 0.05:
                self.object_in_center = True
            # draw the contour and center of the shape on the image TODO: wieder entfernen
            cv.drawContours(img, [contour], -1, (255, 0, 0), 2)
            cv.circle(img, (x, y), 7, (255, 0, 0), -1)
            cv.imwrite('ApproachCenter.jpeg', img)
        else:
            turn_intensity_and_direction = 0.0

        linear = self.param_max_translational_velocity * 0.8
        angular = turn_intensity_and_direction * self.param_max_rotational_velocity

        approach = Twist()
        approach.angular.z = angular
        approach.linear.x = linear
        approach = self.avoid(approach)

        return approach

    def push(self):
        push = Twist()
        robots, robots_center = get_neighbors(self.current_scan, 1.6)
        for robot in robots_center:
            if np.deg2rad(350) < robot[1] < np.deg2rad(10):
                self.publish_info("roboter vor mir! ich bleibe stehen")
                return push
        if get_mean_dist(self.current_scan, -15, 15) > 1.0:
            push.linear.x = self.param_max_translational_velocity
        else:
            push.linear.x = 0.3 * self.param_max_translational_velocity

        return push

    def wall_follow(self):
        """calculates the vector to follow the wall"""
        scan = self.current_scan
        left_mean_dist = get_mean_dist(scan, 30, 80)
        front_mean_dist = get_mean_dist(scan, -30, 30)
        desired_dist = 1.9  # desired distance to the wall
        error = left_mean_dist - desired_dist

        linear = self.param_max_translational_velocity
        angular = error * 1.5

        if front_mean_dist < desired_dist + 0.65:
            # turn right:
            angular = -1.0

        linear, angular = self.scale_velocity(linear, angular)
        wall_follow_direction = Twist()
        wall_follow_direction.angular.z = angular
        wall_follow_direction.linear.x = linear
        wall_follow_direction = self.avoid(wall_follow_direction)

        return wall_follow_direction

    def avoid(self, behavior):
        scan = self.current_scan
        avoid = Twist()
        linear = behavior.linear.x
        angular = behavior.angular.z

        if any(dist < 1.0 for dist in scan.ranges):
            robots, robots_center = get_neighbors(scan, 1.6)
            min_dist = 3.5
            for robot in robots_center:

                if np.deg2rad(270) < robot[1] < np.deg2rad(90) and robot[0] < min_dist:
                    min_dist = robot[0]
            self.publish_info(str(len(robots_center)) + " robots detected!")
            self.publish_info("min_dist = " + str(min_dist))
            if min_dist <= 0.5:
                if get_distance(scan, 180) > 0.7:
                    linear = -0.2
                    angular = 0.0
                else:
                    linear = 0.0
                    angular = 0.0
            elif min_dist < 1.0:
                linear = behavior.linear.x * min_dist * 0.1
                angular = behavior.angular.z * min_dist * 0.6
            elif min_dist < 1.6:
                linear = behavior.linear.x * min_dist * 0.4
                angular = behavior.angular.z * min_dist * 0.4
                linear, angular = self.scale_velocity(linear, angular)

        avoid.linear.x = linear
        avoid.angular.z = angular

        return avoid

    def stop(self):
        stop = Twist()
        self.command_publisher.publish(stop)
        self.publish_info("command published: stop")

    def drive_backwards(self):
        scan = self.current_scan
        if any(dist < 0.8 for dist in scan.ranges):
            self.publish_info("wait")
            backwards = Twist()
            if get_distance(scan, 180) > 0.8:
                backwards.angular.z = 0.0
                backwards.linear.x = - 0.2
            self.command_publisher.publish(backwards)
            self.publish_info("command published: driving backwards")
            time.sleep(1.1)
            self.publish_info("backwards sleep beendet")

    def turn_once(self):
        self.turn_timer = Timer(self.param_turn_timer_period, self.stop)
        turn = Twist()
        turn.angular.z = self.param_max_rotational_velocity
        self.protection.data = 'False'
        self.protection_publisher.publish(self.protection)
        self.command_publisher.publish(turn)
        self.publish_info("command published: turn")
        self.turn_timer.start()

    def is_object_visible(self):
        self.search_object_timer.cancel()
        self.search_object_timer = Timer(self.param_object_timer_period, self.is_object_visible)

        if (not self.turn_timer.is_alive()) and self.state == State.SEARCH:
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

        if not self.search_object_timer.is_alive():
            self.search_object_timer.start()

    def is_goal_occluded(self):
        self.search_goal_timer.cancel()
        self.search_goal_timer = Timer(self.param_goal_timer_period, self.is_goal_occluded)
        if time.time() - self.last_goal_check > 7.0:
            if not (self.state == State.CHECK_FOR_GOAL or self.state == State.PUSH):
                occluded = False
            else:
                occluded = self.goal_is_occluded
                if not self.turn_timer.is_alive():
                    left, right = has_neighbors(self.current_scan)
                    occluded = True
                    self.publish_info("left = " + str(left) + " ,right = " + str(right))

                    if left < 1 or right < 1:
                        self.publish_info("is_goal_occluded? Status ist: " + str(self.state))
                        self.state = State.CHECK_FOR_GOAL
                        self.drive_backwards()
                        self.turn_once()
                        occluded = True
                        for i in range(0, 1000):
                            time.sleep(0.1)
                            self.update_is_goal_in_image()
                            if self.goal_in_image:
                                occluded = False
                            if not self.turn_timer.is_alive():
                                break

            self.goal_is_occluded = occluded
            self.last_goal_check = time.time()
        self.search_goal_timer.start()

    # FLAGS #
    def is_busy(self):
        if (self.turn_timer.is_alive() or (not self.search_object_timer.is_alive()) or (
                not self.search_goal_timer.is_alive())) and self.state != State.INIT:
            return True
        else:
            return False

    def update_is_object_in_image(self):
        image = self.current_image
        lower_color = self.lower_object_color
        upper_color = self.upper_object_color
        self.object_in_image = is_color_in_image(image, lower_color, upper_color)

    def update_is_goal_in_image(self):
        image = self.current_image
        if self.state == State.CHECK_FOR_GOAL:
            cv.imwrite('Goal_In_Image.jpeg', image)
        lower_color = self.lower_goal_color
        upper_color = self.upper_goal_color
        self.goal_in_image = is_color_in_image(image, lower_color, upper_color)

    def update_is_near_object(self):
        old_near_object = self.near_object
        self.near_object = True
        dist_valid = True

        if self.object_in_image:
            robots, robots_center = get_neighbors(self.current_scan, 1.5)
            dist = get_mean_dist(self.current_scan, -10, 10)
            if ((not self.state == State.APPROACH) and dist > 1.7) or dist > 1.5:
                self.near_object = False
            for robot in robots_center:
                if np.deg2rad(350) < robot[1] < np.deg2rad(10):
                    self.publish_info("dist is not valid!")
                    dist_valid = False

        if not dist_valid:
            self.near_object = old_near_object

    # BERECHNUNGEN #
    def scale_velocity(self, translational_velocity, rotational_velocity):
        if abs(translational_velocity) > self.param_max_translational_velocity:
            scale = self.param_max_translational_velocity / abs(translational_velocity)
            translational_velocity = translational_velocity * scale
            rotational_velocity = rotational_velocity * scale
        if abs(rotational_velocity) > self.param_max_rotational_velocity:
            scale = self.param_max_rotational_velocity / abs(rotational_velocity)
            translational_velocity = translational_velocity * scale
            rotational_velocity = rotational_velocity * scale
        return translational_velocity, rotational_velocity

    def update_is_max_transport_time_reached(self):
        if self.transport_start_time is None:
            self.transport_start_time = time.time()

        if time.time() - self.transport_start_time > self.max_transport_time:
            self.max_transport_time_reached = True


def main(args=None):
    """Create a node for the pushing pattern and handle the setup."""
    setup_node.init_and_spin(args, PushingPattern)


if __name__ == '__main__':
    main()
