from kinematics import Kinematics
import numpy as np

# Method 1: Get rx, ry, rz from joint angles
def get_orientation_from_joints(theta1, theta2, theta3, theta4, theta5, theta6):
    # Convert to radians if provided in degrees
    joint_angles = np.array([theta1, theta2, theta3, theta4, theta5, theta6])
    
    # Calculate forward kinematics
    matrix = Kinematics.calculate_forward_kinematics(joint_angles)
    
    # Extract orientation
    _, _, _, rx, ry, rz = Kinematics.from_homogeneous_matrix(matrix)
    return rx, ry, rz

# Method 2: Create a rotation matrix directly
def get_specific_orientation(roll, pitch, yaw):
    # Create homogeneous matrix with zero translation
    matrix = Kinematics.to_homogeneous_matrix(0, 0, 0, roll, pitch, yaw)
    return roll, pitch, yaw  # Just returning the input values

# Example usage
joint_angles = np.array([53.95,-28.66,34.20,34.20,-97,26,-90.82,349.49]) * np.pi / 180
rx, ry, rz = get_orientation_from_joints(joint_angles[0],joint_angles[1],joint_angles[2],joint_angles[3],joint_angles[4],joint_angles[5])
print(f"RX: {rx:.10f}, RY: {ry:.10f}, RZ: {rz:.10f}")