from geometry_msgs.msg import Twist
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

import random


# https://cvexplained.wordpress.com/2020/04/28/color-detection-hsv/

# TODO: eventuell Random walk bewusst nicht implementieren wegen effizienz
# TODO: Verhalten zu den Combined verhalten einordnen
# TODO: Eventuell anstatt is_busy() State Check hinzufügen
# TODO: Move Around Object timer eventuell random in einem Bestimmten bereich setzten
# TODO: Move Around Object right or left wall, abhängig davon machen wann das Ziel gesehen wurde
# TODO: Standard mäßige überprüfung ob Roboter vor einem bei Pushing, dann Protection an, vermutlich zu rechen intensiv
#       * stattdessen lieber implementieren wenn rechts von ihm mehr ist als links, dass er sich leicht in diese richtung dreht
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
    # TODO: eventuell überprüfen ob mehr als eine contour gefunden wurde oder keine
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
    # TODO: effizientere Variante finden
    # center: erster wert entfernung in meter , zweiter wert rad (vorne 0, links rum steigend bis 2pi)
    # roboter werden zwischen 0.2m und 3m erkannt min_range=0, max_range=3, threshold=0.35, min_width=0, max_width=15
    robots, robots_center = ScanCalculationFunctions.identify_robots(laser_scan=scan_msg, min_range=0, max_range=3, threshold=0.35, min_width=0, max_width=15)
    left = 0
    right = 0

    for robot in robots_center:
        if np.deg2rad(110) > robot[1] > np.deg2rad(10):
            left += 1
        elif np.deg2rad(350) > robot[1] > np.deg2rad(250):
            right += 1

    if left > 0 and right > 0:
        return True
    else:
        return False


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
                ('target_search_timer_period', None),
                ('object_search_timer_period', None),
                ('wall_follow_timer_period', None),
                ('max_translational_velocity', None),
                ('max_rotational_velocity', None)
            ])

        self.param_max_translational_velocity = self.get_parameter(
            "max_translational_velocity").get_parameter_value().double_value
        self.param_max_rotational_velocity = self.get_parameter(
            "max_rotational_velocity").get_parameter_value().double_value
        self.param_turn_timer_period = self.get_parameter(
            "turn_timer_period").get_parameter_value().double_value
        self.param_target_timer_period = self.get_parameter(
            "target_search_timer_period").get_parameter_value().double_value
        self.param_object_timer_period = self.get_parameter(
            "object_search_timer_period").get_parameter_value().double_value
        self.param_move_around_object_timer_period = self.get_parameter(
            "wall_follow_timer_period").get_parameter_value().double_value

        # SUBPATTERN #
        self.random_walk_latest = Twist()
        self.random_walk_subpattern = self.create_subscription(
            Twist, self.get_namespace() + '/drive_command_random_walk_pattern', self.command_callback_random_walk, 10)

        # SUBSCRIBER #
        self.scan_subscription = self.create_subscription(LaserScan, self.get_namespace() + '/scan', self.swarm_command_controlled(self.scan_callback), qos_profile=qos_profile_sensor_data)
        self.camera_subscription = self.create_subscription(Image, self.get_namespace() + '/camera/image_raw', self.swarm_command_controlled(self.camera_callback), qos_profile=qos_profile_sensor_data)

        # PUBLISHER #
        self.information_publisher = self.create_publisher(StringMessage, self.get_namespace() + '/information', 10)
        self.protection_publisher = self.create_publisher(StringMessage, self.get_namespace() + '/hardware_protection_layer', 10)

        # GENERAL #
        self.state = State.INIT
        self.current_scan = None
        self.current_image = None
        self.direction = Twist()

        # color in RGB = [76, 142, 24] , in BGR = [24, 142, 76] and in HSV = [47 212 142]
        # [100, 50, 50] bis [140, 255, 255] entspricht rot, aber warum ???
        # [50, 100, 100] bis [70, 255, 255] entspricht grün
        # print(cv.cvtColor(np.uint8([[BGR]]),cv.COLOR_BGR2HSV))

        # OBJEKT #
        self.lower_object_color = np.array([50, 50, 50])  # np.array([50, 50, 50])  # in HSV [H-10, 100,100]
        self.upper_object_color = np.array([70, 255, 255])  # np.array([70, 255, 255])  # in HSV [H+10, 255, 255]
        self.object_in_image = False
        self.near_object = False

        # GOAL #
        self.lower_goal_color = np.array([110, 100, 100])  # np.array([0, 50, 50])  # in HSV [H-10, 100,100]
        self.upper_goal_color = np.array([130, 255, 255])  # np.array([10, 255, 255])  # in HSV [H+10, 255, 255]
        self.goal_in_image = False
        self.goal_is_occluded = False

        self.info = StringMessage()
        self.info.data = 'Starting pushing pattern'
        self.information_publisher.publish(self.info)

        self.protection = StringMessage()
        self.protection.data = 'True'
        self.protection_publisher.publish(self.protection)

        # TIMER #
        self.search_goal_timer = Timer(self.param_target_timer_period, self.is_goal_occluded)
        self.search_object_timer = Timer(self.param_object_timer_period, self.is_object_visible)
        self.turn_timer = Timer(self.param_turn_timer_period, self.stop)
        self.move_around_object_timer = Timer(self.param_move_around_object_timer_period, self.stop)

        # COUNTER #
        self.state_switch_counter = 0

        # TEST und AUSWERTUNG #
        # 0 = time, 1 = velocity, 2 = state, 3 = pose, 4 = object_center, 5 = goal_position]
        self.state_list = [[], [[], []], [], [[], [], []], [[], []], [[], []]]
        self.publish_state_counter = 0
        self.current_object_center = [[], []]
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
        self.state_list[5][0].append(self.goal_position[0])
        self.state_list[5][1].append(self.goal_position[1])
        with open('pushing_state_list_' + str(self.get_namespace())[-1] + '.txt', 'w') as doc:
            doc.write(str(self.state_list))

        self.publish_state_counter -= 1
        if self.publish_state_counter <= 0:
            self.publish_state_counter = 10
            self.info.data = '_____' + str(self.state) + '_____ target=' + str(self.goal_is_occluded) + ' object=' + str(self.object_in_image)
            self.information_publisher.publish(self.info)

            self.info.data = 'BUSY= ' + str(self.is_busy()) + ' turn= ' + str(
                self.turn_timer.is_alive()) + ' goal= ' + str(self.search_goal_timer.is_alive()) + ' object= ' + str(
                self.search_object_timer.is_alive()) + ' wall= ' + str(self.move_around_object_timer.is_alive())
            self.information_publisher.publish(self.info)

    def publish_info(self, msg):
        self.info.data = str(msg)
        self.information_publisher.publish(self.info)

    # CALLBACKS #
    def command_callback_random_walk(self, incoming_msg: Twist):
        """Assign the message to variable"""
        self.random_walk_latest = incoming_msg

    def scan_callback(self, incoming_msg: LaserScan):
        """Call back if a new scan msg is available."""
        self.current_scan = incoming_msg

        # TODO: wieder entfernen, wenn random_walk in launch file wieder einkommentiert ist
        self.random_walk_latest.linear.x = self.param_max_translational_velocity
        self.random_walk_latest.angular.z = random.randint(-1, 2) * random.randint(1, 11) * 0.1 * self.param_max_rotational_velocity

        self.pushing()

    def camera_callback(self, raw_image_msg: Image):
        """Call back if a new image msg is available."""
        self.current_image = bridge.imgmsg_to_cv2(raw_image_msg, desired_encoding='passthrough')

    # STATUS AKTUALISIERUNG UND AUSFÜHRUNG #
    def pushing(self):
        if self.is_busy():
            return
        self.update_flags()
        self.update_state()
        self.publish_state()
        self.execute_state()

    def update_flags(self):
        if self.state != State.INIT:
            self.update_is_object_in_image()
            self.update_is_near_object()

            if self.state == State.PUSH:
                self.update_is_goal_in_image()

    def update_state(self):
        old_state = self.state

        if self.state != State.INIT:
            if self.state == State.MOVE_AROUND_OBJECT:
                if self.move_around_object_timer.is_alive():
                    self.state = State.MOVE_AROUND_OBJECT
                else:
                    self.state = State.SEARCH

            elif not self.object_in_image:
                self.state = State.SEARCH

            elif self.state == State.SEARCH and self.object_in_image:
                self.state = State.APPROACH

            elif self.state == State.APPROACH and self.near_object:
                self.state = State.CHECK_FOR_GOAL

            elif self.state == State.PUSH and self.goal_in_image:
                self.state = State.MOVE_AROUND_OBJECT

            elif self.state == State.CHECK_FOR_GOAL:
                if self.goal_is_occluded:
                    self.state = State.PUSH
                else:
                    self.state = State.MOVE_AROUND_OBJECT
                    self.move_around_object_timer = Timer(self.param_move_around_object_timer_period, self.stop)
                    self.move_around_object_timer.start()

        if self.state == State.SEARCH and old_state != State.SEARCH and old_state != State.MOVE_AROUND_OBJECT and self.state_switch_counter < 20:
            self.state = old_state
            self.state_switch_counter += 1
        else:
            self.state_switch_counter = 0

    def execute_state(self):
        # Führt Verhalten abhängig von Status aus
        if self.state == State.INIT:
            fully_initialized = not (self.current_image is None or self.current_scan is None)
            if fully_initialized:
                self.search_goal_timer.start()
                self.search_object_timer.start()
                self.direction = self.random_walk_latest
                self.state = State.SEARCH

        elif self.state == State.SEARCH:
            self.direction = self.random_walk_latest
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
            self.search_goal_timer = Timer(0, self.is_goal_occluded)
            self.search_goal_timer.start()
            self.protection.data = 'False'

        self.protection_publisher.publish(self.protection)
        self.command_publisher.publish(self.direction)

    # BEWEGUNGSMUSTER #
    def approach(self):
        img = self.current_image
        img_width = img.shape[1]
        contour = get_image_contour(img, self.lower_object_color, self.upper_object_color)
        if contour is not None:
            x, y = get_center(contour)
            # normiert Wert: 1 (ganz links) und -1 (ganz rechts)
            turn_intensity_and_direction = (2 * x / img_width - 1) * -1
            # draw the contour and center of the shape on the image TODO: wieder entfernen
            cv.drawContours(img, [contour], -1, (255, 0, 0), 2)
            cv.circle(img, (x, y), 7, (255, 0, 0), -1)
            cv.imwrite('ApproachCenter.jpeg', img)
        else:
            turn_intensity_and_direction = 0.0

        approach = Twist()
        approach.angular.z = turn_intensity_and_direction * self.param_max_rotational_velocity
        approach.linear.x = self.param_max_translational_velocity

        return approach

    def push(self):
        push = Twist()
        push.angular.z = 0.0  # eventuell abhängig davon, ob links oder rechts eine geringer distanz ist drehen
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

        robots, robots_center = get_neighbors(scan, 1.6)
        min_dist = np.inf
        for robot in robots_center:
            if np.deg2rad(350) < robot[1] < np.deg2rad(10) and robot[0] < min_dist:
                min_dist = robot[0]
            if min_dist <= 1.6:
                linear = linear * min_dist * 0.53

        linear, angular = self.scale_velocity(linear, angular)
        wall_follow_direction = Twist()
        wall_follow_direction.angular.z = angular
        wall_follow_direction.linear.x = linear

        return wall_follow_direction

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
            self.info.data = 'backwards START'
            # self.information_publisher.publish(self.info)
            time.sleep(0.8)

    def turn_once(self):
        self.turn_timer = Timer(self.param_turn_timer_period, self.stop)
        turn = Twist()
        turn.angular.z = self.param_max_rotational_velocity

        self.command_publisher.publish(turn)
        self.turn_timer.start()

        self.info.data = 'turn_timer START'
        # self.information_publisher.publish(self.info)

    def is_object_visible(self):
        self.search_object_timer.cancel()
        self.search_object_timer = Timer(self.param_object_timer_period, self.is_object_visible)

        if (not self.turn_timer.is_alive()) and (self.state == State.SEARCH):
            # self.publish_info('!!!Starte Überprüfung auf OBJEKT!!!')
            self.turn_once()
            for i in range(0, 1000):
                time.sleep(0.1)
                self.update_is_object_in_image()

                if not self.turn_timer.is_alive():
                    # self.publish_info('!!!KEIN OBJEKT in Sicht!!!')
                    break

                elif self.object_in_image:
                    # self.publish_info('!!!OBJEKT in Sicht!!!')
                    self.turn_timer.cancel()
                    self.turn_timer = Timer(self.param_turn_timer_period, self.stop)
                    self.stop()
                    break

        self.search_object_timer.start()

    def is_goal_occluded(self):
        self.search_goal_timer.cancel()
        self.search_goal_timer = Timer(self.param_target_timer_period, self.is_goal_occluded)

        if (not has_neighbors(self.current_scan)) and (not self.turn_timer.is_alive()) and (self.state == State.PUSH or self.state == State.CHECK_FOR_GOAL):
            occluded = True
            self.publish_info('!!!Starte Überprüfung auf ZIEL!!')
            self.drive_backwards()
            self.turn_once()
            for i in range(0, 1000):
                time.sleep(0.1)
                self.update_is_goal_in_image()
                if self.goal_in_image:
                    occluded = False
                    break
                if not self.turn_timer.is_alive():
                    break

            if not occluded:
                self.publish_info('!!!ZIEL in Sicht!!!')
            else:
                self.publish_info('!!!KEIN ZIEL in Sicht!!!')

            self.goal_is_occluded = occluded
        self.search_goal_timer.start()

    # FLAGS #
    def is_busy(self):
        if (self.turn_timer.is_alive() or (not self.search_goal_timer.is_alive()) or (
                not self.search_object_timer.is_alive())) and self.state != State.INIT:
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
            if dist > 1.6:
                self.near_object = False
            for robot in robots_center:
                if np.deg2rad(350) < robot[1] < np.deg2rad(10):
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


def main(args=None):
    """Create a node for the pushing pattern and handle the setup."""
    setup_node.init_and_spin(args, PushingPattern)


if __name__ == '__main__':
    main()
