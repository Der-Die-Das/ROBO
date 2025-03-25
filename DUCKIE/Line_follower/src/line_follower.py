import rospy
import numpy as np
import cv2
import csv
import os
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped
from std_msgs.msg import Header

# Constant definitions (all speeds in metres per second)
KP = 0.2         # Proportional gain (scales current error)
KI = 0.003       # Integral gain (accumulates past errors)
KD = 0.0015         # Derivative gain (predicts future error)

IMAGE_HEIGHT = 0.3
LOWER_YELLOW = np.array([20, 100, 100])
UPPER_YELLOW = np.array([30, 255, 255])

WHEEL_SPEED_MIN = 0.0
WHEEL_SPEED_MAX = 2
MINIMUM_ERROR_THRESHOLD = 0.1
MAXIMUM_ERROR_THRESHOLD = 0.6

LOW_SPEED = 0.4
HIGH_SPEED = 1.0

class LineFollower:
    def __init__(self, robot_name):
        rospy.init_node('linefollower', anonymous=True)
        
        self.last_save_time = rospy.Time.now()
        self.kp = KP
        self.ki = KI
        self.kd = KD

        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = rospy.Time.now()

        self.bridge = CvBridge()
        self.camera_image_msg = CompressedImage()

        self.wheel_pub = rospy.Publisher(f'/{robot_name}/wheels_driver_node/wheels_cmd', WheelsCmdStamped, queue_size=1)
        rospy.Subscriber(f'/{robot_name}/camera_node/image/compressed', CompressedImage, self.image_callback, queue_size=1, buff_size=2**30)

        rospy.on_shutdown(self.stop_robot)
        self.rate = rospy.Rate(20)
        rospy.sleep(2.0)  # Allow time for sensor and topic initialization

        # Variables for lap timing via blue pixel detection
        self.in_lap = False
        self.lap_start_time = None
        self.BLUE_PIXEL_THRESHOLD = 2000  # Adjust this threshold based on your environment

        # Use a class member for blue_mask instead of a global variable.
        self.blue_mask = None

        # Store the program start time and initialize an error log list.
        self.program_start_time = rospy.Time.now()
        self.error_log = []  # Each element: (timestamp, error)

    def run(self):
        last_stamp = self.camera_image_msg.header.stamp
        while not rospy.is_shutdown():
            if self.camera_image_msg.header.stamp != last_stamp:
                # Process lane following control
                vel_left, vel_right = self.analyze_image_street()
                self.set_wheel_speed(vel_left, vel_right)
                # Update lap timing based on blue pixel detection
                self.lap_time_update()
                last_stamp = self.camera_image_msg.header.stamp
            self.rate.sleep()

    def image_callback(self, data):
        self.camera_image_msg = data

    def stop_robot(self):
        self.set_wheel_speed(0.0, 0.0)
        rospy.loginfo("Robot stopped.")
        self.write_error_csv()

    def analyze_image_basic(self):
        image = self.bridge.compressed_imgmsg_to_cv2(self.camera_image_msg, "bgr8")
        height, width, _ = image.shape
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[int(height * (1-IMAGE_HEIGHT)):, :]
        mask = cv2.inRange(hsv_image, LOWER_YELLOW, UPPER_YELLOW)
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        left_half = masked_image[:, :width // 2]
        right_half = masked_image[:, width // 2:]
        left_yellow = cv2.countNonZero(cv2.cvtColor(left_half, cv2.COLOR_BGR2GRAY))
        right_yellow = cv2.countNonZero(cv2.cvtColor(right_half, cv2.COLOR_BGR2GRAY))

        total_yellow = left_yellow + right_yellow
        yellow_ratio = (right_yellow - left_yellow) / total_yellow if total_yellow else 0

        return self.pid_control(yellow_ratio)

    def analyze_image_street(self):
        image = self.bridge.compressed_imgmsg_to_cv2(self.camera_image_msg, "bgr8")
        height, width, _ = image.shape
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[int(height * (1-IMAGE_HEIGHT)):, :]

        yellow_lower_hsv = np.array([20, 100, 100])
        yellow_upper_hsv = np.array([30, 255, 255])
        white_lower_hsv = np.array([0, 0, 150])
        white_upper_hsv = np.array([255, 255, 255])

        yellow_mask = cv2.inRange(hsv_image, yellow_lower_hsv, yellow_upper_hsv)
        white_mask = cv2.inRange(hsv_image, white_lower_hsv, white_upper_hsv)

        yellow_mask[:, width//2:] = 0
        white_mask[:, :width//2] = 0

        white_rows, white_cols = np.where(white_mask > 0)
        avg_x_white = np.mean(white_cols) if white_cols.size > 0 else 9999

        yellow_rows, yellow_cols = np.where(yellow_mask > 0)
        avg_x_yellow = np.mean(yellow_cols) if yellow_cols.size > 0 else 0

        current_time = rospy.Time.now()
        if (current_time - self.last_save_time).to_sec() >= 0.5:
            cv2.imwrite("robot_view.png", image)
            cv2.imwrite("robot_view_white.png", white_mask)
            cv2.imwrite("robot_view_yellow.png", yellow_mask)
            if self.blue_mask is not None:
                cv2.imwrite("robot_view_blue.png", self.blue_mask)
            self.last_save_time = current_time

        white_distance = avg_x_white - width/2
        yellow_distance = width/2 - avg_x_yellow
        error = white_distance - yellow_distance
        error_norm = error / (width * 0.2)

        return self.pid_control(error_norm)

    def pid_control(self, error):
        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()

        self.integral += error * dt
        derivative = (error - self.previous_error) / dt if dt > 0 else 0.0
        pid_output = self.kp * error + self.ki * self.integral + self.kd * derivative

        self.previous_error = error
        self.last_time = current_time

        # Record error along with timestamp (relative to program start)
        rel_time = (current_time - self.program_start_time).to_sec()
        self.error_log.append((rel_time, abs(error)))

        speed = HIGH_SPEED 
        if abs(error) > MINIMUM_ERROR_THRESHOLD and abs(error) < MAXIMUM_ERROR_THRESHOLD:
            speed = HIGH_SPEED - (abs(error) / MAXIMUM_ERROR_THRESHOLD) * (HIGH_SPEED - LOW_SPEED)
        elif abs(error) > MAXIMUM_ERROR_THRESHOLD: 
            speed = LOW_SPEED

        return np.clip(speed + pid_output, WHEEL_SPEED_MIN, WHEEL_SPEED_MAX), np.clip(speed - pid_output, WHEEL_SPEED_MIN, WHEEL_SPEED_MAX)

    def set_wheel_speed(self, left, right):
        wheel_command = WheelsCmdStamped(
            header=Header(stamp=rospy.Time.now()),
            vel_left=left,
            vel_right=right
        )
        self.wheel_pub.publish(wheel_command)

    def lap_time_update(self):
        # Crop the image to the lower 20%
        image = self.bridge.compressed_imgmsg_to_cv2(self.camera_image_msg, "bgr8")
        height, width, _ = image.shape
        lower_img = image[int(height * 0.8):, :]
        hsv_lower = cv2.cvtColor(lower_img, cv2.COLOR_BGR2HSV)

        # Define HSV thresholds for blue detection in the lower 20%
        lower_blue = np.array([100, 150, 0])
        upper_blue = np.array([140, 255, 255])
        self.blue_mask = cv2.inRange(hsv_lower, lower_blue, upper_blue)

        blue_count = cv2.countNonZero(self.blue_mask)
        current_blue_detected = blue_count > self.BLUE_PIXEL_THRESHOLD

        if not hasattr(self, 'prev_blue_detected'):
            self.prev_blue_detected = current_blue_detected

        # Transition: blue was seen and now it's gone.
        if self.prev_blue_detected and not current_blue_detected:
            if not self.in_lap:
                self.lap_start_time = rospy.Time.now()
                self.in_lap = True
                rospy.loginfo("Lap started.")
            else:
                lap_time = (rospy.Time.now() - self.lap_start_time).to_sec()
                rospy.loginfo(f"Lap time: {lap_time:.2f} seconds.")
                self.append_lap_data(lap_time)
                self.lap_start_time = rospy.Time.now()
        self.prev_blue_detected = current_blue_detected

    def append_lap_data(self, lap_time):
        csv_file = "lap_times.csv"
        row = [
            rospy.Time.now().to_sec(),
            lap_time,
            self.kp,
            self.ki,
            self.kd,
            self.BLUE_PIXEL_THRESHOLD,
            MINIMUM_ERROR_THRESHOLD,
            MAXIMUM_ERROR_THRESHOLD,
            LOW_SPEED,
            HIGH_SPEED,
        ]
        write_header = not os.path.isfile(csv_file) or os.stat(csv_file).st_size == 0
        with open(csv_file, "a") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "timestamp", "lap_time", "kp", "ki", "kd", "BLUE_PIXEL_THRESHOLD",
                    "MINIMUM_ERROR_THRESHOLD", "MAXIMUM_ERROR_THRESHOLD", "LOW_SPEED", "HIGH_SPEED"
                ])
            writer.writerow(row)

    def write_error_csv(self):
        csv_file = "errors.csv"
        with open(csv_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "error"])
            for t, e in self.error_log:
                writer.writerow([t, e])

if __name__ == '__main__':
    follower = LineFollower("phi")
    follower.run()
