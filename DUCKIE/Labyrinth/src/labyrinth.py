import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped
from std_msgs.msg import Header
from tape_maze_detector import *
from tremaux_solver import *

class RobotController:
    def __init__(self):
        rospy.init_node("maze_solver_node")
        self.rate = rospy.Rate(60)  # 60 Hz
        
        # Get robot name (default: "robot0")
        self.robot_name = "rho"
        
        # Subscribers & Publisher
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(f"/{self.robot_name}/camera_node/image/compressed",
                                            CompressedImage, self.image_callback)
        self.wheels_pub = rospy.Publisher(f"/{self.robot_name}/wheels_driver_node/wheels_cmd",
                                          WheelsCmdStamped, queue_size=10)
        
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
            self.current_frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {}".format(e))

    def publish_wheel_velocity(self, left_vel, right_vel):
        cmd = WheelsCmdStamped()
        cmd.header = Header()
        cmd.header.stamp = rospy.Time.now()
        # Assuming WheelsCmdStamped has these fields for left and right velocities:
        cmd.vel_left = left_vel
        cmd.vel_right = right_vel
        self.wheels_pub.publish(cmd)
        
    def follow_line(self):
        """
        Dummy line-following behavior. In a real robot, this would be replaced
        by a control algorithm that follows the red tape.
        """
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
                self.stop_robot()
                rospy.loginfo("Junction detected with directions: {}".format(available_dirs))
                if available_dirs:
                    chosen_direction = self.solver.junction_reached(self.fixed_distance, available_dirs)
                    rospy.loginfo("Solver recommends to go {}".format(chosen_direction))
                    self.detector.update_heading(chosen_direction)
                else:
                    rospy.logwarn("Junction detected but no valid directions found.")
            self.rate.sleep()

if __name__ == "__main__":
    try:
        controller = RobotController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
