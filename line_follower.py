#!/usr/bin/python3

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped
from std_msgs.msg import Header

# Constant definitions (constants are immutable parameters during runtime)
KP = 0.2         # Proportional gain (scales current error)
KI = 0.005  # 0.005# Integral gain (accumulates past errors)
KD = 0.0                   # Derivative gain (predicts future error)

# HSV boundaries for detecting yellow (used in image segmentation)
IMAGE_HEIGHT = 0.3
LOWER_YELLOW = np.array([20, 100, 100])
UPPER_YELLOW = np.array([30, 255, 255])

# Speed constraints and thresholds (all speeds in metres per second)
WHEEL_SPEED_MIN = 0.1
WHEEL_SPEED_MAX = 0.6
HIGH_ERROR_THRESHOLD = 0.4
LOW_SPEED = 0.3
HIGH_SPEED = 0.5

class LineFollower:
    def __init__(self, robot_name):
        # Initialize the ROS node
        rospy.init_node('linefollower', anonymous=True)
        
        self.last_save_time = rospy.Time.now()
        # Initialize PID parameters and internal state variables
        self.kp = KP
        self.ki = KI
        self.kd = KD

        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = rospy.Time.now()

        # Setup CV bridge for image conversion and store incoming camera images
        self.bridge = CvBridge()
        self.camera_image_msg = CompressedImage()

        # Create publisher for wheel commands and subscriber for camera images
        self.wheel_pub = rospy.Publisher(f'/{robot_name}/wheels_driver_node/wheels_cmd', WheelsCmdStamped, queue_size=1)
        rospy.Subscriber(f'/{robot_name}/camera_node/image/compressed', CompressedImage, self.image_callback, queue_size=1, buff_size=2**30)

        # Define shutdown behavior and set a consistent loop rate
        rospy.on_shutdown(self.stop_robot)
        self.rate = rospy.Rate(20)
        rospy.sleep(2.0)  # Allow time for sensor and topic initialization

    def run(self):
        last_stamp = self.camera_image_msg.header.stamp
        # Main control loop: update and process new images to control wheel speeds
        while not rospy.is_shutdown():
            if self.camera_image_msg.header.stamp != last_stamp:
                # vel_left, vel_right = self.analyze_image_basic()
                vel_left, vel_right = self.analyze_image_street()
                self.set_wheel_speed(vel_left, vel_right)
                last_stamp = self.camera_image_msg.header.stamp
            self.rate.sleep()

    def image_callback(self, data):
        # Update the latest camera image
        self.camera_image_msg = data

    def stop_robot(self):
        # Gracefully stop the robot when shutting down
        self.set_wheel_speed(0.0, 0.0)
        rospy.loginfo("Robot stopped.")

    def analyze_image_basic(self):
        # Convert the compressed ROS image to an OpenCV-compatible format
        image = self.bridge.compressed_imgmsg_to_cv2(self.camera_image_msg, "bgr8")
        # Transform the image from BGR to HSV color space (HSV: Hue, Saturation, Value)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[int(height * (1-IMAGE_HEIGHT)):, :]


        # Generate a binary mask where yellow pixels fall within defined HSV thresholds
        mask = cv2.inRange(hsv_image, LOWER_YELLOW, UPPER_YELLOW)
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        height, width, _ = image.shape
        # Focus on the lower part of the image (assumed to be the region where the line appears)

        # Divide the region into left and right halves for analysis
        left_half = masked_image[:, :width // 2]
        right_half = masked_image[:, width // 2:]

        # Count the number of yellow pixels in each half
        left_yellow = cv2.countNonZero(cv2.cvtColor(left_half, cv2.COLOR_BGR2GRAY))
        right_yellow = cv2.countNonZero(cv2.cvtColor(right_half, cv2.COLOR_BGR2GRAY))

        total_yellow = left_yellow + right_yellow
        # Calculate the error ratio based on the distribution of yellow pixels
        yellow_ratio = (right_yellow - left_yellow) / total_yellow if total_yellow else 0

        rospy.loginfo(f"Yellow balance ratio: {yellow_ratio:.2f}")
        return self.pid_control(yellow_ratio)

    def analyze_image_street(self):
        # Convert the compressed ROS image to an OpenCV-compatible format
        image = self.bridge.compressed_imgmsg_to_cv2(self.camera_image_msg, "bgr8")
        height, width, _ = image.shape

        # Convert image from BGR to HSV color space (HSV: Hue, Saturation, Value)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[int(height * (1-IMAGE_HEIGHT)):, :]

        # Define HSV thresholds for yellow and white detection
        yellow_lower_hsv = np.array([20, 100, 100])   # Lower bound for yellow hue
        yellow_upper_hsv = np.array([30, 255, 255])     # Upper bound for yellow hue
        white_lower_hsv = np.array([0, 0, 150])         # Lower bound for white: minimal saturation, high brightness
        white_upper_hsv = np.array([255, 255, 255])       # Upper bound for white: any hue with low saturation

        # Generate masks for yellow and white based on their HSV thresholds
        yellow_mask = cv2.inRange(hsv_image, yellow_lower_hsv, yellow_upper_hsv)
        white_mask = cv2.inRange(hsv_image, white_lower_hsv, white_upper_hsv)

        # Filter masks to detect yellow only on the left half and white only on the right half
        yellow_mask[:, width//2:] = 0    # Remove yellow detections on the right side
        white_mask[:, :width//2] = 0       # Remove white detections on the left side

        # Compute the average x-coordinate for white pixels on the right side
        white_rows, white_cols = np.where(white_mask > 0)
        avg_x_white = np.mean(white_cols) if white_cols.size > 0 else 9999

        # Compute the average x-coordinate for yellow pixels on the left side
        yellow_rows, yellow_cols = np.where(yellow_mask > 0)
        avg_x_yellow = np.mean(yellow_cols) if yellow_cols.size > 0 else 0

        # Optionally save images for debugging
        current_time = rospy.Time.now()
        if (current_time - self.last_save_time).to_sec() >= 0.5:
            cv2.imwrite("robot_view.png", image)
            cv2.imwrite("robot_view_white.png", white_mask)
            cv2.imwrite("robot_view_yellow.png", yellow_mask)
            self.last_save_time = current_time

        # Calculate distances from image center (global coordinates)
        # rospy.loginfo(f"White avg x: {avg_x_white:.2f}, Yellow avg x: {avg_x_yellow:.2f}")
        white_distance = avg_x_white - width/2
        yellow_distance = width/2 - avg_x_yellow
        # Compute error as the difference of these distances
        error = white_distance - yellow_distance
        error_norm = error / (width * 0.2)
        # rospy.loginfo(f"Normalized error: {error_norm:.2f}")
        # print(f"White distance: {white_distance:.2f}, Yellow distance: {yellow_distance:.2f} error norm: {error_norm:.2f}")

        return self.pid_control(error_norm)



    def pid_control(self, error):
        # Determine elapsed time for PID calculation (dt: time difference)
        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()

        # Calculate the integral (cumulative error) and derivative (rate of error change)
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt if dt > 0 else 0.0

        # Compute PID output using the formula: P + I + D
        pid_output = self.kp * error + self.ki * self.integral + self.kd * derivative

        # Update the error tracking variables for the next cycle
        self.previous_error = error
        self.last_time = current_time

        # Adjust speed based on error magnitude: slower for high error (improves stability)
        speed = LOW_SPEED if abs(error) > HIGH_ERROR_THRESHOLD else HIGH_SPEED

        # Compute individual wheel speeds and constrain them within allowed limits
        vel_left = np.clip(speed + pid_output, WHEEL_SPEED_MIN, WHEEL_SPEED_MAX)
        vel_right = np.clip(speed - pid_output, WHEEL_SPEED_MIN, WHEEL_SPEED_MAX)

        rospy.loginfo(f"PID output: {pid_output:.2f}, Left speed: {vel_left:.2f}, Right speed: {vel_right:.2f}")
        return vel_left, vel_right

    def set_wheel_speed(self, left, right):
        # Publish the computed wheel speeds to the appropriate ROS topic
        wheel_command = WheelsCmdStamped(
            header=Header(stamp=rospy.Time.now()),
            vel_left=left,
            vel_right=right
        )
        self.wheel_pub.publish(wheel_command)

if __name__ == '__main__':
    follower = LineFollower("phi")
    follower.run()
