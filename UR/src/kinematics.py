#! /usr/bin/env python3

import numpy as np


class Kinematics:
    # Denavit-Hartenberg parameters (a, d, alpha) from https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/
    DH_PARAMS = [
        (0, 0.15185, np.pi / 2),
        (-0.24355, 0, 0),
        (-0.2132, 0, 0),
        (0, 0.13105, np.pi / 2),
        (0, 0.08535, -np.pi / 2),
        (0, 0.0921, 0)
    ]

    @classmethod
    def calculate_forward_kinematics(cls, joint_angles):
        """ Returns a 4x4 homogeneous matrix that corresponds to the pose of the end effector """
        # Values are from https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/
        return (
                cls.dh_matrix(joint_angles[0], *cls.DH_PARAMS[0])
                @ cls.dh_matrix(joint_angles[1], *cls.DH_PARAMS[1])
                @ cls.dh_matrix(joint_angles[2], *cls.DH_PARAMS[2])
                @ cls.dh_matrix(joint_angles[3], *cls.DH_PARAMS[3])
                @ cls.dh_matrix(joint_angles[4], *cls.DH_PARAMS[4])
                @ cls.dh_matrix(joint_angles[5], *cls.DH_PARAMS[5])
                # @ cls.dh_matrix(0, 0, 0.15, 0)
        )

    @classmethod
    def calculate_inverse_kinematics(cls, x, y, z, rx, ry, rz, initial_guess=None, max_iterations=1_000, tolerance=1e-3):
        """ Calculates angles of 6 joints from target position (x, y, z, rx, ry and rz) with Newton-Raphson """
        if initial_guess is not None:
            joint_angles = np.copy(initial_guess)
        else:
            joint_angles = np.array([-np.pi / 2, -np.pi * 3 / 4, -np.pi / 4, -np.pi / 2, np.pi / 2, 0], dtype=np.float64)

        target_position = np.array([x, y, z, rx, ry, rz], dtype=np.float64)

        for i in range(max_iterations):
            # Calculate current error
            e = cls.error_function(joint_angles, target_position)
            error_norm = np.linalg.norm(e)
            print(f"Iteration {i+1}: Error = {error_norm:.6f}")

            # Return angles if error is low and therefore approximation close to solution was found
            if error_norm < tolerance:
                return np.mod(joint_angles + np.pi, 2 * np.pi) - np.pi

            # Calculate jacobian matrix
            J = cls.jacobian(joint_angles, target_position)

            # Caclculate the delta of the joint angles to approximate
            delta_angles = np.linalg.pinv(J) @ e
            joint_angles -= delta_angles

        if initial_guess is not None:
            return cls.calculate_inverse_kinematics(x, y, z, rx, ry, rz)

        raise RuntimeError("Inverse Kinematics did not converge")

    @classmethod
    def jacobian(cls, joint_angles, target_position, delta=1e-5):
        """ Numerical calculation of jacobian matrix """
        n = len(joint_angles)
        e = cls.error_function(joint_angles, target_position)
        J = np.zeros((n, n))

        for i in range(n):
            # Change one joint angle by small delta
            joint_angles_perturbed = np.copy(joint_angles)
            joint_angles_perturbed[i] += delta

            # Fill row of current joint in jacobian matrix based on error after perturbation of angle
            J[:, i] = (cls.error_function(joint_angles_perturbed, target_position) - e) / delta

        return J

    @classmethod
    def error_function(cls, joint_angles, target_position):
        """ Calculates errors in x, y, z, rx, ry and rz based on joint angles """
        end_effector_pose = cls.calculate_forward_kinematics(joint_angles)
        x, y, z, rx, ry, rz = cls.from_homogeneous_matrix(end_effector_pose)
        return np.array([x, y, z, rx, ry, rz]) - target_position

    @classmethod
    def dh_matrix(cls, theta, a, d, alpha):
        """ Builds matrix for Denavit-Hartenberg transformation """
        return np.array([
            [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
            [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])

    @classmethod
    def from_homogeneous_matrix(cls, matrix):
        """ Returns x, y, z, rx, ry and rz from homogeneous matrix """
        rx = np.arctan2(matrix[2, 1], matrix[2, 2])
        ry = np.arctan2(-matrix[2, 0], np.sqrt(matrix[2, 1] ** 2 + matrix[2, 2] ** 2))
        rz = np.arctan2(matrix[1, 0], matrix[0, 0])
        return matrix[0, 3], matrix[1, 3], matrix[2, 3], rx, ry, rz

    @classmethod
    def to_homogeneous_matrix(cls, x, y, z, rx, ry, rz):
        """ Builds homogeneous matrix from x, y, z, rx, ry and rz """
        return np.array([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ]) @ np.array([
            [np.cos(rz), -np.sin(rz), 0, 0],
            [np.sin(rz), np.cos(rz), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]) @ np.array([
            [np.cos(ry), 0, np.sin(ry), 0],
            [0, 1, 0, 0],
            [-np.sin(ry), 0, np.cos(ry), 0],
            [0, 0, 0, 1],
        ]) @ np.array([
            [1, 0, 0, 0],
            [0, np.cos(rx), -np.sin(rx), 0],
            [0, np.sin(rx), np.cos(rx), 0],
            [0, 0, 0, 1]
        ])
