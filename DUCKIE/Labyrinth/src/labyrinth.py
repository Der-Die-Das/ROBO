import rospy
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped
from std_msgs.msg import Header
from tape_maze_detector import *

import rospy

class RobotController:
    def __init__(self):
        rospy.init_node("maze_solver_node")
        self.rate = rospy.Rate(60)  # 60 Hz
        
        # Subscribers & Publishers
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.image_callback)
        self.left_wheel_pub = rospy.Publisher("/left_wheel_vel", Float32, queue_size=10)
        self.right_wheel_pub = rospy.Publisher("/right_wheel_vel", Float32, queue_size=10)
        
        # Instantiate our detector and solver.
        self.detector = TapeMazeDetector(current_heading="north")
        self.solver = TremauxSolver(threshold=2.0)
        
        # For simplicity, assume a fixed distance between junctions.
        self.fixed_distance = 5.0  # metres
        
        # Control state
        self.junction_mode = False  # currently processing a junction
        
        # Latest image frame
        self.current_frame = None

    def image_callback(self, msg):
        try:
            self.current_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {}".format(e))

    def publish_wheel_velocity(self, left_vel, right_vel):
        self.left_wheel_pub.publish(Float32(left_vel))
        self.right_wheel_pub.publish(Float32(right_vel))
        
    def follow_line(self):
        """
        Dummy line-following behavior. In a real robot, this would be replaced
        by a control algorithm that follows the red tape.
        """
        # Here we simply command a forward velocity.
        forward_speed = 0.2  # m/s
        self.publish_wheel_velocity(forward_speed, forward_speed)

    def stop_robot(self):
        self.publish_wheel_velocity(0.0, 0.0)

    def run(self):
        rospy.loginfo("Maze solver node started.")
        while not rospy.is_shutdown():
            if self.current_frame is None:
                self.rate.sleep()
                continue
            
            # Run line-following until a junction is detected.
            junction_detected, available_dirs = self.detector.process_image(self.current_frame)
            if not junction_detected:
                self.follow_line()
            else:
                # Stop before processing the junction.
                self.stop_robot()
                rospy.loginfo("Junction detected with directions: {}".format(available_dirs))
                if available_dirs:
                    chosen_direction = self.solver.junction_reached(self.fixed_distance, available_dirs)
                    rospy.loginfo("Solver recommends to go {}".format(chosen_direction))
                    # Update detector heading to reflect new direction.
                    self.detector.update_heading(chosen_direction)
                    # In a real robot, execute a turn here.
                    # For simulation purposes, you might add a delay or a dedicated turn routine.
                else:
                    rospy.logwarn("Junction detected but no valid directions found.")
            self.rate.sleep()

if __name__ == "__main__":
    try:
        controller = RobotController()
        controller.run()
    except rospy.ROSInterruptException:
        pass