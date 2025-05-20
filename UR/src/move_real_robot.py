#! /usr/bin/env python3

import rospy
import socket
from kinematics import Kinematics


class RealRobotArm:

    def __init__(self, host="192.168.1.11"):
        self.last_joint_angles = None

        host = host
        port_ur = 30002
        port_gripper = 63352

        rospy.init_node('my_real_robot')
        rospy.sleep(3.0)        
        # Create socket connection to robot arm and gripper
        self.socket_ur = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_ur.connect((host, port_ur))
        self.socket_gripper = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_gripper.connect((host, port_gripper))
        # activate the gripper
        self.socket_gripper.sendall(b'SET ACT 1\n')

    def grab_object(self, x, y, z, rx, ry, rz, gripper=255):
        """ Moves over object, then down and grabs with gripper and then back up """
        # Move over object and wait till completion
        joint_angles_over = Kinematics.calculate_inverse_kinematics(x=x, y=y, z=z + 0.03, rx=rx, ry=ry, rz=rz, initial_guess=self.last_joint_angles)
        self.send_joint_command(joint_angles_over)
        rospy.sleep(4)

        # Lower down over object and wait till completion
        joint_angles_grab = Kinematics.calculate_inverse_kinematics(x=x, y=y, z=z, rx=rx, ry=ry, rz=rz, initial_guess=joint_angles_over)
        self.send_joint_command(joint_angles_grab)
        rospy.sleep(1)

        # Close/Open gripper after reaching object and wait till completion
        self.send_gripper_command(gripper)
        rospy.sleep(2)

        # Move back up and wait till completion
        self.send_joint_command(joint_angles_over)
        rospy.sleep(1)

    def send_joint_command(self, joint_angles):
        self.last_joint_angles = joint_angles
        values = ', '.join(['{:.2f}'.format(i) if type(i) == float else str(i) for i in joint_angles])
        self.socket_ur.send (str.encode("movej(["+ values + "])\n"))

    def send_gripper_command(self, value):
        if (value >= 0 and value <= 255):
            command = 'SET POS ' + str(value) + '\n'
            self.socket_gripper.send(str.encode(command))
            # make the gripper move
            self.socket_gripper.send(b'SET GTO 1\n')

    def close_connection(self):
        self.socket_ur.close()
        self.socket_gripper.close()


if __name__ == '__main__':
    robot = RealRobotArm()

    # Open gripper on start
    robot.send_gripper_command(0)

    try:

        # Grab first object
        robot.grab_object(x=-0.256, y=-0.355, z=0.178, rx=3.1324254952, ry=0.0179405314, rz=2.6869307545, gripper=120)# (x=-0.008, y=-0.445, z=0.171, rx=-3.1075944832382634, ry=0.02368545818219514, rz=-3.138161729708561, gripper=150)
        
        # Place first object
        robot.grab_object(x=-0.010, y=-0.514, z=0.178, rx=3.0998241501, ry=0.0694890250, rz=-3.1108098793, gripper=0)# (x=-0.008, y=-0.445, z=0.171, rx=-3.1075944832382634, ry=0.02368545818219514, rz=-3.138161729708561, gripper=150)

        # # Grab second object 
        # robot.send_joint_command([0,-1.5708,0,-1.5708,0,0])

        robot.grab_object(x=-0.256, y=-0.355, z=0.178, rx=3.1324254952, ry=0.0179405314, rz=2.6869307545, gripper=120)# (x=-0.008, y=-0.445, z=0.171, rx=-3.1075944832382634, ry=0.02368545818219514, rz=-3.138161729708561, gripper=150)
        # robot.grab_object(x=-0.196, y=-0.492, z=0.168, rx=0.8520325788, ry=-0.2703414087, rz=2.9002713219, gripper=120)# (x=-0.008, y=-0.445, z=0.171, rx=-3.1075944832382634, ry=0.02368545818219514, rz=-3.138161729708561, gripper=150)

        # # Place second object on top of first object
        robot.grab_object(x=-0.010, y=-0.514, z=0.198, rx=3.0998241501, ry=0.0694890250, rz=-2.7037555298, gripper=0)# (x=-0.008, y=-0.445, z=0.171, rx=-3.1075944832382634, ry=0.02368545818219514, rz=-3.138161729708561, gripper=150)
        
        
    except RuntimeError as e:
        rospy.logerr(str(e))
        # Open gripper on error
        robot.send_gripper_command(0)

    # Move back to upright position
    joint_angles = [0, -1.57, 0, -1.57, 0, 0]
    robot.send_joint_command(joint_angles)

    robot.close_connection()
