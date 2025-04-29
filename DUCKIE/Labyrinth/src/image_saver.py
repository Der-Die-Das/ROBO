import rospy
import cv2
import os
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
import numpy as np # Keep numpy as cv_bridge might implicitly depend on it

# Define the interval for saving images (in seconds)
SAVE_INTERVAL = 0.5  # Save an image every 0.5 seconds
# Define the output filename (will be overwritten)
OUTPUT_FILENAME = "robot_view_latest.png"

class ImageSaver:
    def __init__(self, robot_name):
        # Initialize the ROS node
        rospy.init_node('image_saver_node', anonymous=True)
        rospy.loginfo(f"Image Saver node started for robot: {robot_name}")

        # Initialize the time when the last image was saved
        # Set to 0 initially to ensure the first received image triggers a save
        self.last_save_time = rospy.Time(0)
        self.save_interval_duration = rospy.Duration(SAVE_INTERVAL)

        # Create a CvBridge object to convert ROS images to OpenCV format
        self.bridge = CvBridge()

        # Define the output path for the image
        # You might want to make this an absolute path or configurable
        self.output_path = OUTPUT_FILENAME
        rospy.loginfo(f"Images will be saved to: {os.path.abspath(self.output_path)}")

        # Subscribe to the compressed image topic from the robot's camera
        # Increased buff_size might be needed if images arrive faster than processed
        rospy.Subscriber(
            f'/{robot_name}/camera_node/image/compressed',
            CompressedImage,
            self.image_callback,
            queue_size=1,  # Keep only the latest image
            buff_size=2**24 # Buffer size (adjust if needed)
        )

        # Register a shutdown hook if needed (optional for this simple task)
        # rospy.on_shutdown(self.shutdown_hook)

    def image_callback(self, data):
        """
        This function is called every time a new image message is received.
        It checks if enough time has passed since the last save and saves
        the current image if the interval has been met, overwriting the old file.
        """
        current_time = rospy.Time.now()

        # Check if the specified interval has passed since the last save
        if (current_time - self.last_save_time) >= self.save_interval_duration:
            rospy.logdebug("Save interval met. Processing image.") # Use logdebug for potentially frequent messages
            try:
                # Convert the ROS CompressedImage message to an OpenCV image (BGR format)
                cv_image = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")

                # Save the OpenCV image to the specified file path
                # cv2.imwrite will overwrite the file if it already exists
                success = cv2.imwrite(self.output_path, cv_image)

                if success:
                    # Update the last save time only if save was successful
                    self.last_save_time = current_time
                    rospy.loginfo(f"Image saved to {self.output_path} at time {current_time.to_sec()}")
                else:
                    rospy.logwarn(f"Failed to write image to {self.output_path}")

            except Exception as e:
                # Log any errors during conversion or saving
                rospy.logerr(f"Error processing image: {e}")
        else:
            # Optional: Log if image is skipped due to interval not met
            rospy.logdebug("Skipping image save (interval not met).")
            pass

    # def shutdown_hook(self):
    #     """Optional: Called when ROS node is shutting down."""
    #     rospy.loginfo("Image Saver node shutting down.")
        # Add any cleanup tasks here if needed

if __name__ == '__main__':
    # Replace "phi" with the actual name of your Duckiebot if different
    robot_name = "rho"
    try:
        # Create an instance of the ImageSaver class
        saver = ImageSaver(robot_name)
        # Keep the node running until it's shut down (e.g., by Ctrl+C)
        rospy.spin()
    except rospy.ROSInterruptException:
        # Handle the case where the node is interrupted
        rospy.loginfo("Image Saver node interrupted and shutting down.")
    except Exception as e:
        rospy.logfatal(f"An unexpected error occurred: {e}")