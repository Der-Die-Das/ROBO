#! /usr/bin/env python3

import rospy
import numpy as np
import math
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from urdf_parser_py.urdf import URDF
from kinematics import Kinematics



class ForwardKinematics:

    def __init__(self):  
        # we create two publishers in our ROS node
        # one to publish the angles of the joints (joint states)
        # and another for visualizing a target pose (to see whether your calculations are correct)
        self.joint_state_publisher = rospy.Publisher('my_joint_states', JointState, queue_size=10) 
        self.pose_publisher = rospy.Publisher('my_pose', PoseStamped, queue_size=10)
        rospy.init_node('my_joint_state_publisher')
        self.rate = rospy.Rate(10)
        rospy.sleep(3.0)


    def run(self):
        # to get the kinematic chain with the joints and the corresponding parameters, we use the urdf parser
        # documentation of urdf_parser_py see http://wiki.ros.org/urdfdom_py
        robot = URDF.from_parameter_server()
        root = robot.get_root()
        tip = "tool0"
        joint_names = robot.get_chain(root, tip, joints=True, links=False, fixed=False)
        # the properties of a given joint / link can be obtained with the joint_map
        # see http://wiki.ros.org/urdf/XML/joint

        joint_angles = [0,-np.pi/2,0,-np.pi/2,0,0] # in radians


        # publish the joint state values and the target pose
        # create the joint state messages
        js = JointState()
        js.name = joint_names      
        js.position = joint_angles

        end_effector_pose = self.calculate_forward_kinematics(joint_angles, joint_names, robot)
        
        if end_effector_pose is not None :
            target_pose_message = self.get_pose_message_from_matrix(end_effector_pose)
        else:
            rospy.logerr("error, no target pose calculated, use identity matrix instead")
            target_pose_message = self.get_pose_message_from_matrix(np.identity(4))
        while not rospy.is_shutdown():
            self.joint_state_publisher.publish(js)
            self.pose_publisher.publish(target_pose_message)
            self.rate.sleep()

 
    def calculate_forward_kinematics(self, joint_positions, joint_names, robot):
        return Kinematics.calculate_forward_kinematics(joint_positions)

        # # to implement, should return a 4x4 homogeneous matrix that corresponds to the pose of the end effector
        # dh_params_list = [
        #     {'a': 0,        'd': 0.15185, 'alpha': np.pi/2},
        #     {'a': -0.24355, 'd': 0,       'alpha': -np.pi/2}, #0
        #     {'a': -0.2132,  'd': 0,       'alpha': 0},
        #     # {'a': 0,        'd': 0.13105, 'alpha': np.pi/2}, #np.pi/2
        #     # {'a': 0,        'd': 0.08535, 'alpha': -np.pi/2},
        #     # {'a': 0,        'd': 0.0921,  'alpha': 0}
        # ]
        # T_cumulative = np.identity(4) # Start with identity matrix (base frame)

        # for i in range(len(dh_params_list)):
        #     theta_i = joint_positions[i] # Assuming DH table theta offset is 0
        #     a_i = dh_params_list[i]['a']
        #     d_i = dh_params_list[i]['d']
        #     alpha_i = dh_params_list[i]['alpha']

        #     T_rot_z = self.get_rot_z_matrix(theta_i)
        #     T_trans_z = self.get_trans_z_matrix(d_i)
        #     T_trans_x = self.get_trans_x_matrix(a_i)
        #     T_rot_x = self.get_rot_x_matrix(alpha_i)

        #     # Standard DH convention: RotZ * TransZ * TransX * RotX
        #     T_link = T_rot_z @ T_trans_z @ T_trans_x @ T_rot_x
        #     # Or if using Craig's convention (RotX_prev * TransX_prev * RotZ * TransZ), adapt accordingly.
        #     # Your table looks like standard/modified DH.

        #     T_cumulative = T_cumulative @ T_link

        # return T_cumulative
        # initial = np.identity(4)
        # a = self.rotate_and_translate(initial,
        #     0,
        #     joint_positions[0],
        #     0,
        #     0.15185)
        # b = self.rotate_and_translate(a,
        #     joint_positions[1]-3.142/2,
        #     0,
        #     -0.24355,
        #     0)
        # c = self.rotate_and_translate(b,
        #     joint_positions[2],
        #     0,
        #     0,
        #     0.2132)
        # a = self.rotate(initial, joint_positions[0])
        # b = self.translate(a, 0.15185)
        # c = self.rotate(b, joint_positions[1], -3.142/2)
        # d = self.translate(c, 0.24355)
        # e = self.rotate(d, joint_positions[2])
        # f = self.translate(e, 0.2132)
        # g = self.rotate(f, joint_positions[3],-3.142/2)
        # h = self.translate(g, 0.13105)
        # i = self.rotate(h, joint_positions[4])
        # j = self.translate(i, 0.08535)
        # k = self.rotate(j, joint_positions[5])
        # l = self.translate(k, 0.0921)
        # return l

    def rotate_and_translate(self,input, theta_x, theta_z, x, z):
        cz = np.cos(theta_z)
        sz = np.sin(theta_z)
        cx = np.cos(theta_x)
        sx = np.sin(theta_x)

        matrix =  np.array([
            [ cz, -sz * cx,  sz * sx, x ],
            [ sz,  cz * cx, -cz * sx, 0 ],
            [  0,      sx,      cx,   z ],
            [  0,       0,       0,   1 ]
        ])
        return np.dot(input, matrix)    
    def rotate(self,input, theta, offset=0):
        rospy.loginfo(f"Offset: {offset}")
        theta_new = theta+offset
        rotation_matrix = T = np.array([
            [np.cos(theta_new), -np.sin(theta_new),     0,  0],
            [np.sin(theta_new),  np.cos(theta_new),     0,  0],
            [0,             0,          1,          0],
            [0,             0,          0,          1]
        ])
        return np.dot(input, rotation_matrix)

    def translate(self,input, x,y,z):
        translation_matrix = T = np.array([
            [1,             0,          0,          x],
            [0,             1,          0,          y],
            [0,             0,          1,          z],
            [0,             0,          0,          1]
        ])
        return np.dot(input, translation_matrix)
    def get_rot_z_matrix(self,theta):
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([
            [c, -s, 0, 0],
            [s,  c, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1]
        ])

    def get_trans_z_matrix(self,d):
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, d],
            [0, 0, 0, 1]
        ])

    def get_trans_x_matrix(self,a):
        return np.array([
            [1, 0, 0, a],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def get_rot_x_matrix(self,alpha):
        c = np.cos(alpha)
        s = np.sin(alpha)
        return np.array([
            [1, 0,  0, 0],
            [0, c, -s, 0],
            [0, s,  c, 0],
            [0, 0,  0, 1]
        ])

    # the following function creates a PoseStamped message from a homogeneous matrix
    def get_pose_message_from_matrix(self, matrix):

        """Return pose msgs from homogeneous matrix
        matrix : homogeneous matrix 4x4
        """
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "base"
        pose = pose_stamped.pose
        pose.position.x = matrix[0][3]
        pose.position.y = matrix[1][3]
        pose.position.z = matrix[2][3]

        q = self.get_quaternion_from_matrix(matrix)
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
        return pose_stamped

    # the ROS message type PoseStamped uses quaternions for the orientation
    def get_quaternion_from_matrix(self, matrix):
        """Return quaternion from homogeneous matrix
        matrix : homogeneous matrix 4x4
        """
        q = np.empty((4,), dtype=np.float64)
        M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
        t = np.trace(M)
        if t > M[3, 3]:
            q[3] = t
            q[2] = M[1, 0] - M[0, 1]
            q[1] = M[0, 2] - M[2, 0]
            q[0] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
        return q


if __name__ == '__main__':
    fk = ForwardKinematics()
    fk.run()