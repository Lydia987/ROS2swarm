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
# TODO: effizientere Methode finden Nachbarn zu erkennen
# TODO: Verhalten zu den Combined verhalten einordnen
# TODO: Eventuell anstatt is_busy() State Check hinzufügen
# TODO: Bei ersten mal PUSH_OBJECT sofort auf Ziel überprüfen
# TODO: Move Around Object timer eventuell random in einem Bestimmten bereich setzten
# TODO: Move Around Object right or left wall, abhängig davon machen wann das Ziel gesehen wurde
# TODO: get neighbours beachten das mind. ein roboter links daneben und einer rechts daneben
# TODO: Umbenennung der Klasse und aktualisierung + schreiben aller Komentare


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
                ('object_distance_reached', None),
                ('max_translational_velocity', None),
                ('max_rotational_velocity', None)
            ])

        self.param_max_translational_velocity = self.get_parameter(
            "max_translational_velocity").get_parameter_value().double_value
        self.param_max_rotational_velocity = self.get_parameter(
            "max_rotational_velocity").get_parameter_value().double_value
        self.param_distance_reached = self.get_parameter(
            "object_distance_reached").get_parameter_value().double_value
        self.param_turn_timer_period = self.get_parameter(
            "turn_timer_period").get_parameter_value().double_value
        self.param_target_timer_period = self.get_parameter(
            "target_search_timer_period").get_parameter_value().double_value
        self.param_object_timer_period = self.get_parameter(
            "object_search_timer_period").get_parameter_value().double_value
        self.param_move_around_object_timer_period = self.get_parameter(
            "wall_follow_timer_period").get_parameter_value().double_value

        # TODO: Timer wieder einkommentieren, aktuell start über scanner callback
        # self.timer = self.create_timer(
        #     self.param_goal_timer_period,
        #     self.swarm_command_controlled_timer(self.caging))

        self.random_walk_latest = Twist()
        # messages from subpattern subscription
        self.random_walk_subpattern = self.create_subscription(
            Twist,
            self.get_namespace() + '/drive_command_random_walk_pattern',
            self.command_callback_random_walk,
            10)

        self.scan_subscription = self.create_subscription(
            LaserScan,
            self.get_namespace() + '/scan',
            self.swarm_command_controlled(self.scan_callback),
            qos_profile=qos_profile_sensor_data
        )

        self.camera_subscription = self.create_subscription(
            Image,
            self.get_namespace() + '/camera/image_raw',
            self.swarm_command_controlled(self.camera_callback),
            qos_profile=qos_profile_sensor_data
        )

        self.information_publisher = self.create_publisher(
            StringMessage,
            self.get_namespace() + '/information',
            10
        )

        self.protection_publisher = self.create_publisher(
            StringMessage,
            self.get_namespace() + '/hardware_protection_layer',
            10
        )

        self.state = State.INIT
        self.current_scan = None
        self.current_image = None
        self.direction = Twist()

        # color in RGB = [76, 142, 24] , in BGR = [24, 142, 76] and in HSV = [47 212 142]
        # [100, 50, 50] bis [140, 255, 255] entspricht rot, aber warum ???
        # [50, 100, 100] bis [70, 255, 255] entspricht grün
        # print(cv.cvtColor(np.uint8([[BGR]]),cv.COLOR_BGR2HSV))
        self.lower_object_color = np.array([50, 50, 50])  # np.array([50, 50, 50])  # in HSV [H-10, 100,100]
        self.upper_object_color = np.array([70, 255, 255])  # np.array([70, 255, 255])  # in HSV [H+10, 255, 255]
        self.lower_goal_color = np.array([110, 100, 100])  # np.array([0, 50, 50])  # in HSV [H-10, 100,100]
        self.upper_goal_color = np.array([130, 255, 255])  # np.array([10, 255, 255])  # in HSV [H+10, 255, 255]
        self.object_in_image = False
        self.object_direction = None
        self.object_distance = np.inf
        self.goal_is_occluded = False

        self.info = StringMessage()
        self.info.data = 'Starting pushing pattern'
        self.information_publisher.publish(self.info)

        self.protection = StringMessage()
        self.protection.data = 'True'
        self.protection_publisher.publish(self.protection)

        # self.state_timer = Timer(0.01, self.caging)
        # self.state_timer.start()
        self.search_goal_timer = Timer(self.param_target_timer_period, self.is_goal_occluded,
                                       args=(self.current_scan, self.lower_goal_color, self.upper_goal_color))
        self.search_object_timer = Timer(self.param_object_timer_period, self.is_object_visible,
                                         args=(self.lower_object_color, self.upper_object_color))
        self.turn_timer = Timer(self.param_turn_timer_period, self.stop)
        self.move_around_object_timer = Timer(self.param_move_around_object_timer_period, self.stop)

        self.counter = 0
        self.switch_counter = 0
        self.old_busy = [False, True, True, False]

    def stop(self):
        stop = Twist()
        self.command_publisher.publish(stop)

    def command_callback_random_walk(self, incoming_msg: Twist):
        """Assign the message to variable"""
        self.random_walk_latest = incoming_msg

    def scan_callback(self, incoming_msg: LaserScan):
        """Call back if a new scan msg is available."""
        # self.info.data = "Bin im Scanner callback"
        # self.information_publisher.publish(self.info)
        self.current_scan = incoming_msg

        # TODO: wieder entfernen, wenn random_walk in launch file wieder einkommentiert ist
        self.random_walk_latest.linear.x = self.param_max_translational_velocity
        self.random_walk_latest.angular.z = random.randint(-1, 2) * random.randint(1, 11) * 0.1 * self.param_max_rotational_velocity

        # TODO: wieder entfernen, wenn timer einkommentiert
        # self.pushing()

    def camera_callback(self, raw_image_msg: Image):
        """Call back if a new scan msg is available."""
        # self.info.data = "Bin im Camera callback"
        # self.information_publisher.publish(self.info)
        self.current_image = bridge.imgmsg_to_cv2(raw_image_msg, desired_encoding='passthrough')

    def print_timer_state(self):
        self.info.data = 'BUSY= ' + str(self.is_busy()) + ' turn= ' + str(
            self.turn_timer.is_alive()) + ' target= ' + str(self.search_goal_timer.is_alive()) + ' object= ' + str(
            self.search_object_timer.is_alive()) + ' wall= ' + str(self.move_around_object_timer.is_alive())
        # self.information_publisher.publish(self.info)

    def pushing(self):
        old_state = self.state
        # self.info.data = 'Bin in Timer Callback'
        # self.information_publisher.publish(self.info)
        if self.is_busy():
            new_busy = [self.turn_timer.is_alive(), self.search_goal_timer.is_alive(),
                        self.search_object_timer.is_alive(), self.move_around_object_timer.is_alive()]
            if self.old_busy != new_busy:
                self.print_timer_state()
                self.old_busy = new_busy
            # self.state_timer = Timer(1.0, self.caging)
            # self.state_timer.start()
            return
        else:
            self.update_state()

        if self.state == State.SEARCH_OBJECT and old_state != State.SEARCH_OBJECT and old_state != State.MOVE_AROUND_OBJECT and self.switch_counter < 20:
            self.state = old_state
            self.switch_counter += 1
        else:
            self.switch_counter = 0

        self.counter -= 1
        if self.counter <= 0:
            self.counter = 10
            self.info.data = '_____' + str(self.state) + '_____ target=' + str(self.goal_is_occluded) + ' object=' + str(
                self.object_in_image)  # + ' dist=' + str(self.object_distance)
            # self.information_publisher.publish(self.info)

        self.execute_state()

        # self.state_timer = Timer(1.0, self.caging)
        # self.state_timer.start()

    def update_state(self):
        if self.state != State.INIT:
            self.object_in_image, self.object_direction, self.object_distance = self.get_object_infos(
                self.current_scan, self.current_image, self.lower_object_color, self.upper_object_color)

            if self.state == State.MOVE_AROUND_OBJECT:
                if self.move_around_object_timer.is_alive():
                    self.state = State.MOVE_AROUND_OBJECT
                else:
                    self.state = State.SEARCH_OBJECT

            elif not self.object_in_image:
                self.state = State.SEARCH_OBJECT

            elif self.state == State.SEARCH_OBJECT and self.object_in_image:
                self.state = State.APPROACH_OBJECT

            elif self.state == State.APPROACH_OBJECT and self.object_distance <= self.param_distance_reached:
                self.state = State.CHECK_FOR_GOAL

            elif self.state == State.CHECK_FOR_GOAL:
                if self.goal_is_occluded:
                    self.state = State.PUSH_OBJECT
                else:
                    self.state = State.MOVE_AROUND_OBJECT
                    self.move_around_object_timer = Timer(self.param_move_around_object_timer_period, self.stop)
                    self.move_around_object_timer.start()

    def execute_state(self):
        # Führt Verhalten abhängig von Status aus
        if self.state == State.INIT:
            fully_initialized = not (self.current_image is None or self.current_scan is None)
            if fully_initialized:
                self.search_goal_timer.start()
                self.search_object_timer.start()
                self.direction = self.random_walk_latest
                self.state = State.SEARCH_OBJECT

        elif self.state == State.SEARCH_OBJECT:
            self.direction = self.random_walk_latest
            self.protection.data = 'True'

        elif self.state == State.APPROACH_OBJECT:
            self.direction.angular.z = self.object_direction
            self.direction.linear.x = self.param_max_translational_velocity
            self.protection.data = 'True'

        elif self.state == State.PUSH_OBJECT:
            self.direction.angular.z = self.object_direction
            self.direction.linear.x = 0.5 * self.param_max_translational_velocity
            self.protection.data = 'False'

        elif self.state == State.MOVE_AROUND_OBJECT:
            self.direction = self.wall_follow(self.current_scan)
            self.protection.data = 'False'

        elif self.state == State.CHECK_FOR_GOAL:
            self.search_goal_timer.cancel()
            self.search_goal_timer = Timer(0, self.is_goal_occluded, args=(self.current_scan, self.lower_goal_color, self.upper_goal_color))
            self.search_goal_timer.start()
            self.protection.data = 'False'

        # self.info.data = 'Direction: lin=' + str(self.direction.linear.x) + ' rot=' + str(self.direction.angular.z)
        # self.information_publisher.publish(self.info)
        self.protection_publisher.publish(self.protection)
        self.command_publisher.publish(self.direction)

    def is_busy(self):
        if (self.turn_timer.is_alive() or (not self.search_goal_timer.is_alive()) or (
                not self.search_object_timer.is_alive())) and self.state != State.INIT:
            return True
        else:
            return False

    def has_neighbors(self, scan_msg):
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

    def is_color_in_image(self, image, lower_color, upper_color):
        # self.info.data = 'is_color_in_image'
        # self.information_publisher.publish(self.info)
        is_in_image = False
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        obj_mask = cv.inRange(hsv, lower_color, upper_color)
        number_of_white_pix = np.sum(obj_mask == 255)
        number_of_pix = obj_mask.shape[0] * obj_mask.shape[1]
        percentage_of_white_pix = number_of_white_pix * (100 / number_of_pix)
        if percentage_of_white_pix > 0.01:
            is_in_image = True
        return is_in_image

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
            time.sleep(1.2)  # TODO: param daraus machen der in der Yaml datei steht

    def turn_once(self):
        self.turn_timer = Timer(self.param_turn_timer_period, self.stop)
        turn = Twist()
        turn.angular.z = self.param_max_rotational_velocity

        self.command_publisher.publish(turn)
        self.turn_timer.start()

        self.info.data = 'turn_timer START'
        # self.information_publisher.publish(self.info)

    def is_object_visible(self, lower_color, upper_color):
        self.search_object_timer.cancel()
        self.search_object_timer = Timer(self.param_object_timer_period, self.is_object_visible,
                                         args=(self.lower_object_color, self.upper_object_color))

        in_image = self.object_in_image

        if (not self.turn_timer.is_alive()) and self.state == State.SEARCH_OBJECT:
            self.info.data = '!!!Starte Überprüfung auf OBJEKT!!!'
            # self.information_publisher.publish(self.info)
            self.turn_once()
            for i in range(0, 1000):
                time.sleep(0.1)
                in_image = self.is_color_in_image(self.current_image, lower_color, upper_color)
                # self.info.data = 'Objekt Bild ' + str(i)
                # self.information_publisher.publish(self.info)

                if not self.turn_timer.is_alive():
                    self.info.data = '!!!KEIN OBJEKT in Sicht!!!'
                    # self.information_publisher.publish(self.info)

                    in_image = False
                    break

                elif in_image:
                    self.info.data = '!!!OBJEKT in Sicht!!!'
                    # self.information_publisher.publish(self.info)

                    in_image = True
                    self.turn_timer.cancel()
                    self.turn_timer = Timer(self.param_turn_timer_period, self.stop)
                    self.info.data = 'Der turn_timer wurde gecancelt'
                    # self.information_publisher.publish(self.info)
                    self.stop()
                    break

        self.object_in_image = in_image

        self.search_object_timer.start()

    def is_goal_occluded(self, scan_msg, lower_color, upper_color):
        self.search_goal_timer.cancel()
        self.search_goal_timer = Timer(self.param_target_timer_period, self.is_goal_occluded,
                                       args=(self.current_scan, self.lower_goal_color, self.upper_goal_color))

        if self.has_neighbors(scan_msg) or self.state != State.PUSH_OBJECT or self.state != State.CHECK_FOR_GOAL:
            occluded = True
        else:
            occluded = self.goal_is_occluded

        if (not self.turn_timer.is_alive()) and (not self.has_neighbors(scan_msg)) and ((self.state == State.PUSH_OBJECT) or (self.state == State.CHECK_FOR_GOAL)):
            self.info.data = '!!!Starte Überprüfung auf ZIEL!!'
            # self.information_publisher.publish(self.info)
            self.drive_backwards()
            self.turn_once()
            for i in range(0, 1000):
                time.sleep(0.1)
                # self.info.data = 'Ziel Bild ' + str(i)
                # self.information_publisher.publish(self.info)
                occluded = not self.is_color_in_image(self.current_image, lower_color, upper_color)
                if (not self.turn_timer.is_alive()) or (not occluded):
                    break

            if not occluded:
                self.info.data = '!!!ZIEL in Sicht!!!'
                # self.information_publisher.publish(self.info)
            else:
                self.info.data = '!!!KEIN ZIEL in Sicht!!!'
                # self.information_publisher.publish(self.info)

        self.goal_is_occluded = occluded

        self.search_goal_timer.start()

    def get_object_infos(self, scan_msg, image, lower_color, upper_color):
        is_in_image = self.is_color_in_image(image, lower_color, upper_color)
        obj_distance = np.inf
        obj_direction = 0.0

        if is_in_image:
            hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
            obj_mask = cv.inRange(hsv, lower_color, upper_color)
            x, y = self.get_centroid(obj_mask)

            cv.imwrite('Original.jpeg', image)
            cv.imwrite('Maske.jpeg', obj_mask)

            # normiert Wert: 1 (ganz links) und -1 (ganz rechts)
            img_width = image.shape[1]
            turn_intensity_and_direction = (2 * x / img_width - 1) * -1

            # Camera module v 2.1: Horizontal field of view = 62.2 deg & Vertical field of view = 48.8 deg
            centroid_degree = turn_intensity_and_direction * 31.1

            obj_distance = self.get_distance(scan_msg, centroid_degree)
            obj_direction = turn_intensity_and_direction * self.param_max_rotational_velocity

            # self.info.data = 'relative Centroid: x=' + str(x / img_width) + '  y=' + str(
            #     y / img_width) + '  centroid_degree= ' + str(centroid_degree) + '  Distanz=' + str(obj_distance)
            # self.information_publisher.publish(self.info)

        return is_in_image, obj_direction, obj_distance

    def get_centroid(self, img):
        x = 0
        y = 0
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)  # Nur damit Kontur in anderer Farbe eingezeichnet werden kann
        grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        median = cv.medianBlur(grey, 5)
        blurred = cv.GaussianBlur(median, (5, 5), 0)
        thresh = cv.threshold(blurred, 60, 255, cv.THRESH_BINARY)[1]

        # find contours in the thresholded image
        cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        it = 0
        # loop over the contours
        for c in cnts:
            it += 1
            # compute the center of the contour
            M = cv.moments(c)
            if M["m00"] != 0:
                x = int(M["m10"] / M["m00"])
                y = int(M["m01"] / M["m00"])
            # draw the contour and center of the shape on the image
            cv.drawContours(img, [c], -1, (255, 0, 0), 2)
            cv.circle(img, (x, y), 7, (255, 0, 0), -1)
            cv.putText(img, str(it), (x - 50, y - 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # TODO: Bild speichern wieder entfernen
        cv.imwrite('Thresh.jpeg', thresh)
        cv.imwrite('Center.jpeg', img)

        return x, y

    def wall_follow(self, scan_msg):
        """calculates the vector to follow the wall"""

        desired_dist = 2  # desired distance to the wall
        correction_strength = 0.25

        wall_follow_direction = Twist()
        wall_follow_direction.linear.x = self.param_max_translational_velocity

        left_mean_dist = self.get_mean_dist(scan_msg, -80, -30)
        front_mean_dist = self.get_mean_dist(scan_msg, -30, 30)

        # Follow left wall
        wall_follow_direction.angular.z = 1 * correction_strength * (desired_dist - left_mean_dist)
        # self.info.data = 'desired_dist-left_mean_dist' + str(desired_dist - left_mean_dist)
        # self.information_publisher.publish(self.info)
        if front_mean_dist < desired_dist + 0.4:
            # turn right:
            wall_follow_direction.angular.z = -0.9 * self.param_max_rotational_velocity
        elif front_mean_dist > 1.8 and left_mean_dist > 2.0:
            wall_follow_direction.angular.z = 0.5 * self.param_max_rotational_velocity

        # self.info.data = 'Direction: lin=' + str(wall_follow_direction.linear.x) + ' rot=' + str(
        #     wall_follow_direction.angular.z)
        # self.information_publisher.publish(self.info)

        return wall_follow_direction

    def get_distance(self, scan_msg, degree):
        rad = np.deg2rad(degree)
        increment = scan_msg.angle_increment
        min_rad = scan_msg.angle_min
        max_rad = scan_msg.angle_max
        if rad < min_rad or rad > max_rad:
            distance = np.inf
        else:
            distance = scan_msg.ranges[int((rad - min_rad) / increment)]
        return distance

    def get_mean_dist(self, scan_msg, min_angle, max_angle):
        """ calculates the mean distance to obstacles between the min_angle and max_angle """
        R = 2.5  # repulsive force up to R in meter
        sum_dist = 0.0
        start_angle = np.deg2rad(min_angle)  # rad (front = 0 rad)
        end_angle = np.deg2rad(max_angle)  # rad (front = 0 rad)

        if start_angle < scan_msg.angle_min:
            start_angle = scan_msg.angle_min
        if end_angle > scan_msg.angle_max:
            end_angle = scan_msg.angle_max

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


def main(args=None):
    """Create a node for the pushing pattern and handle the setup."""
    setup_node.init_and_spin(args, PushingPattern)


if __name__ == '__main__':
    main()
