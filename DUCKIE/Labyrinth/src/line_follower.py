# src/maze_solver_pkg/line_follower.py
import cv2
import rospy # For logging

class SimpleLineFollower:
    """
    A simple line follower using image moments for basic proportional control.
    Calculates a desired angular velocity based on line position in the image.
    """
    def __init__(self, kp, img_width):
        self.kp = kp                  # Proportional gain
        self.img_width = img_width    # Width of the input image for centering calculation
        self.width_percentage = 0.8
        self.height_percentage = 0.3
        self.image_center_x = self.img_width*self.width_percentage / 2.0

    def calculate_angular_velocity(self, frame):
        """
        Calculates desired angular velocity based on the line's centroid.

        Args:
            frame: The BGR image frame (or relevant ROI).

        Returns:
            A tuple: (angular_velocity_cmd, line_detected).
                     angular_velocity_cmd is the suggested angular velocity (float).
                     line_detected is True if a line was found, False otherwise.
        """
        if frame is None:
            rospy.logwarn_throttle(2.0, "LineFollower: No frame provided.")
            return 0.0, False

        h, w, _ = frame.shape
        if w != self.img_width:
             # Handle potential resizing or unexpected frame width
             rospy.logwarn_throttle(5.0, f"LineFollower: Frame width ({w}) differs from expected ({self.img_width}). Recalculating center.")
             self.image_center_x = w / 2.0
             self.img_width = w # Update for consistency within this call


        # Example: find white pixels centroid in bottom part of image
        # TODO: This mask and ROI might need tuning based on TapeMazeDetector's needs
        w_1 = int((self.width_percentage//2)* self.img_width)
        w_2 = int((1-(self.width_percentage//2))* self.img_width)
        h_1 = int((self.height_percentage)* h)
        roi = frame[h//2:, w_1:w_2] # Look at bottom half
        mask = cv2.inRange(roi, (150, 150, 150), (255, 255, 255)) # Simple white mask

        M = cv2.moments(mask)
        if M["m00"] > 0:
            line_center_x = int(M["m10"] / M["m00"])
            error = line_center_x - self.image_center_x
            normalized_error = error / self.image_center_x # Normalize error (e.g., to [-1, 1])
            angular_vel_cmd = -self.kp * normalized_error # Negative sign depends on control convention

            # Optional: Clamp angular velocity?
            # max_angular = 1.0 # Example limit
            # angular_vel_cmd = max(-max_angular, min(max_angular, angular_vel_cmd))

            return angular_vel_cmd, True
        else:
            # No line detected in ROI
            return 0.0, False