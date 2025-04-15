# src/maze_solver_pkg/robot_controller.py
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped
import time # Using Python time for simple turn duration

# Import local modules (assuming they are in the same Python package)
from state import State
from tape_maze_detector import TapeMazeDetector # Assuming your detector class is here
from tremaux_solver import TremauxSolver       # Assuming your solver class is here
from line_follower import SimpleLineFollower   # Import the simple follower


class RobotController:
    def __init__(self):
        # Node initialization moved to the main executable script
        self.rate = rospy.Rate(50)  # Hz
        rospy.on_shutdown(self.shutdown_hook)

        # --- Parameters ---
        self.robot_name = rospy.get_param("~robot_name", "rho")
        # Following Params
        self.fwd_speed_following = rospy.get_param("~fwd_speed_following", 0.06)
        kp_line_follow = rospy.get_param("~kp_line_follow", 5.0) # Gain for simple follower
        # Turning Params
        self.turn_speed_angular = rospy.get_param("~turn_speed_angular", 5) # rad/s
        self.turn_duration_90_deg = rospy.get_param("~turn_duration_90_deg", 1.8) # Seconds (NEEDS TUNING)
        self.kinematics_wheel_base = rospy.get_param("~kinematics_wheel_base", 0.102) # meters, for diff drive conversion
        # Solver Params
        self.segment_distance_assumption = rospy.get_param("~segment_distance_assumption", 0.3) # metres
        self.tremaux_junction_threshold = rospy.get_param("~tremaux_junction_threshold", 0.1) # metres
        # Detector Params
        detector_border_crop = rospy.get_param("~detector_border_crop", 10)
        detector_angle_tol = rospy.get_param("~detector_angle_tol", 30.0)
        detector_forward_thresh = rospy.get_param("~detector_forward_thresh", 60)
        detector_mask_crop = rospy.get_param("~detector_mask_crop", 40.0)
        detector_mask_thresh = rospy.get_param("~detector_mask_thresh", 120)

        # --- State Machine ---
        self.current_state = State.FOLLOWING
        self.target_heading = None # Store the heading we want to turn towards
        self.pre_turn_heading = None # Store the heading *before* the turn decision
        self.turn_start_time = None
        self.effective_turn_duration = 0.0 # Store calculated duration for the current turn
        self.last_state_change = rospy.Time(0)

        # --- ROS Comms ---
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            f"/{self.robot_name}/camera_node/image/compressed",
            CompressedImage, self.image_callback, queue_size=1, buff_size=2**30
        )
        self.wheels_pub = rospy.Publisher(
            f"/{self.robot_name}/wheels_driver_node/wheels_cmd",
            WheelsCmdStamped, queue_size=1
        )

        # --- Core Logic Components ---
        # Assuming image width is needed by follower (e.g., 640x480) - parameterize this?
        # Let's get it from the first image received, or assume a default
        self.image_width = rospy.get_param("~camera_image_width", 640) # Default width
        self.line_follower = SimpleLineFollower(kp=kp_line_follow, img_width=self.image_width)

        self.detector = TapeMazeDetector(
            current_heading="north", # Initial assumption
            border_crop_pixels=detector_border_crop,
            angle_tolerance=detector_angle_tol,
            forward_top_threshold_px=detector_forward_thresh,
            mask_crop_percentage=detector_mask_crop,
            mask_lower_threshold=detector_mask_thresh,
            # Add other detector params as needed
        )
        self.solver = TremauxSolver(threshold=self.tremaux_junction_threshold)

        # --- Data ---
        self.current_frame = None
        self.last_image_time = rospy.Time(0)

        rospy.loginfo(f"[{rospy.get_name()}] Initialized. State: {self.current_state.name}")
        rospy.loginfo(f"[{rospy.get_name()}] Following Speed: {self.fwd_speed_following:.2f}, Turn Speed: {self.turn_speed_angular:.2f} rad/s")
        rospy.loginfo(f"[{rospy.get_name()}] Solver Dist Assumption: {self.segment_distance_assumption:.2f}m, Junction Threshold: {self.tremaux_junction_threshold:.2f}m")
        rospy.loginfo(f"[{rospy.get_name()}] Detector Initial Heading: {self.detector.current_heading}")

    def image_callback(self, msg):
        """Handles incoming compressed image messages."""
        try:
            self.current_frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            self.last_image_time = msg.header.stamp
            # Update image width if needed (only on first valid frame)
            if self.image_width != self.current_frame.shape[1] and self.current_frame is not None:
                 rospy.loginfo(f"Updating image width for line follower from {self.image_width} to {self.current_frame.shape[1]}")
                 self.image_width = self.current_frame.shape[1]
                 self.line_follower.img_width = self.image_width
                 self.line_follower.image_center_x = self.image_width / 2.0

        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            self.current_frame = None
        except Exception as e:
            rospy.logerr(f"Error processing compressed image: {e}")
            self.current_frame = None

    def publish_wheel_velocity(self, left_vel, right_vel):
        """Publishes direct wheel velocities."""
        cmd = WheelsCmdStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.vel_left = float(left_vel)
        cmd.vel_right = float(right_vel)
        self.wheels_pub.publish(cmd)

    def convert_twist_to_wheels(self, linear_vel, angular_vel):
        """Converts linear/angular velocity to differential drive wheel velocities."""
        # Basic differential drive kinematics:
        # v_left = linear_vel - (angular_vel * wheel_base / 2)
        # v_right = linear_vel + (angular_vel * wheel_base / 2)
        # Assumes wheel velocities are in m/s (or same units as linear_vel)
        left_vel = linear_vel - (angular_vel * self.kinematics_wheel_base / 2.0)
        right_vel = linear_vel + (angular_vel * self.kinematics_wheel_base / 2.0)
        return left_vel, right_vel

    def stop_robot(self):
        """Sends command to stop the robot wheels."""
        self.publish_wheel_velocity(0.0, 0.0)
        rospy.loginfo_throttle(2.0, "Robot stop command sent.")

    def shutdown_hook(self):
        """Ensures robot stops when ROS node shuts down."""
        rospy.logwarn(f"[{rospy.get_name()}] Shutting down. Stopping robot.")
        self.stop_robot()
        rospy.sleep(0.5) # Give publisher time
        self.stop_robot() # Send stop again

    # --- State Handlers ---

    def handle_following_state(self):
        """Logic for the FOLLOWING state: Detect junctions and follow line."""
        if self.current_frame is None:
            return

        # 1. Process Image for Junction Detection
        self.junction_detected = False
        self.available_dirs = []
        try:
            # Detector needs the robot's current heading (updated after turns)
            self.junction_detected, self.available_dirs = self.detector.process_image(self.current_frame)
            if self.junction_detected:
                rospy.loginfo(f"Detector (H: {self.detector.current_heading}): Junc={self.junction_detected}, Dirs={self.available_dirs}")
        except Exception as e:
             rospy.logerr(f"Error during tape detection: {e}")
             # Consider stopping or going to ERROR state on detector failure
             self.stop_robot()
             self.current_state = State.ERROR
             return

        # 2. Check for State Transition
        last_state_change_time = (rospy.Time.now() - self.last_state_change).to_sec()
        if self.junction_detected:
            rospy.loginfo(f"last state change: {last_state_change_time}")
        if self.junction_detected and (last_state_change_time > 2):
            rospy.loginfo(f"Transition: {self.current_state.name} -> {State.STOPPED_AT_JUNCTION.name}")
            self.current_state = State.STOPPED_AT_JUNCTION
            self.last_state_change = rospy.Time.now()
            self.stop_robot() # Stop immediately upon detecting junction
            return 
        
        # 3. Execute Action: Follow Line using SimpleLineFollower
        angular_vel_cmd, line_detected = self.line_follower.calculate_angular_velocity(self.current_frame)

        if line_detected:
            left_vel, right_vel = self.convert_twist_to_wheels(self.fwd_speed_following, angular_vel_cmd)
            self.publish_wheel_velocity(left_vel, right_vel)
            rospy.loginfo_throttle(2.0, f"Following: Lin={self.fwd_speed_following:.2f}, Ang={angular_vel_cmd:.2f} -> L={left_vel:.2f}, R={right_vel:.2f}")
        else:
            # Line lost during following
            rospy.logwarn_throttle(1.0,"Line lost during FOLLOWING state. Stopping.")
            self.stop_robot()
            # Optional: Could enter an ERROR state or attempt recovery

    def handle_stopped_at_junction_state(self):
        """Logic for STOPPED_AT_JUNCTION: Confirm detection, call solver, decide next state."""
        rospy.loginfo_once("Entered STOPPED_AT_JUNCTION state.")

        if self.current_frame is None:
             rospy.logwarn("STOPPED_AT_JUNCTION: No image frame available to confirm directions. Waiting.")
             # Robot is already stopped, just wait for a frame
             return

        # 1. Re-process image to confirm available directions
        # 2. Make Decision using Solver
        if not self.available_dirs:
            rospy.logwarn("STOPPED_AT_JUNCTION: No available directions detected despite junction flag. Treating as dead end.")
            # Tremaux logic for dead end: Turn around (180 degrees)
            # Determine direction came from using solver's memory
            came_from = self.solver.opposites.get(self.solver.last_direction) if self.solver.last_direction else None

            if came_from:
                 rospy.logwarn(f"Planning 180 turn back towards {came_from}.")
                 self.pre_turn_heading = self.detector.current_heading
                 self.target_heading = came_from # Target is where we entered from
                 # Update detector's heading *now* to reflect state *after* the turn
                 self.detector.update_heading(self.target_heading)
                 rospy.loginfo(f"Detector heading updated to: {self.detector.current_heading}")
                 self.current_state = State.TURNING
                 self.last_state_change = rospy.Time.now()
                 self.turn_start_time = None # Reset turn timer flag
            else:
                 rospy.logerr("STOPPED_AT_JUNCTION: Dead end detected, but cannot determine entry direction! Entering ERROR state.")
                 self.current_state = State.ERROR
            return # Exit state handler

        # Valid directions available, call solver
        try:
            # Pass the assumed distance travelled since last junction
            chosen_direction = self.solver.junction_reached(self.segment_distance_assumption, self.available_dirs)
            rospy.loginfo(f"Solver recommends: {chosen_direction} (Current heading: {self.detector.current_heading})")
        except Exception as e:
            rospy.logerr(f"Error calling Tremaux solver: {e}")
            self.current_state = State.ERROR
            return

        # 3. Decide Next Action Based on Solver Choice
        if chosen_direction == self.detector.current_heading:
             rospy.loginfo("Chosen direction is current heading. Proceeding forward.")
             rospy.loginfo(f"Transition: {self.current_state.name} -> {State.FOLLOWING.name}")
             self.current_state = State.FOLLOWING
             self.last_state_change = rospy.Time.now()
             # No turn needed, detector heading remains the same.
        else:
             # Prepare for turning
             self.pre_turn_heading = self.detector.current_heading
             self.target_heading = chosen_direction
             rospy.loginfo(f"Transition: {self.current_state.name} -> {State.TURNING.name} (From: {self.pre_turn_heading}, Target: {self.target_heading})")
             self.current_state = State.TURNING
             self.last_state_change = rospy.Time.now()
             self.turn_start_time = None # Reset turn timer flag
             # Update detector's heading *now* to reflect the state *after* the turn completes
             self.detector.update_heading(self.target_heading)
             rospy.loginfo(f"Detector heading updated to: {self.detector.current_heading} (for next segment)")


    def handle_turning_state(self):
        """Logic for the TURNING state: Execute timed turn using wheel velocities."""
        if self.target_heading is None or self.pre_turn_heading is None:
            rospy.logerr("TURNING state invalid: target or previous heading missing! Returning to FOLLOWING.")
            self.current_state = State.FOLLOWING
            self.last_state_change = rospy.Time.now()
            self.stop_robot()
            return

        current_time = rospy.Time.now()

        # Start Turn Timer & Action (if not started)
        if self.turn_start_time is None:
            rospy.loginfo(f"Starting turn from {self.pre_turn_heading} towards {self.target_heading}...")
            self.turn_start_time = current_time

            # Calculate required angle difference (0, 90, 180, 270 degrees)
            current_angle = self.detector.cardinal_map.get(self.pre_turn_heading, 0)
            target_angle = self.detector.cardinal_map.get(self.target_heading, 0)
            angle_diff = (target_angle - current_angle + 360) % 360
            rospy.loginfo(f"Required turn angle: {angle_diff} degrees")

            # Calculate duration multiplier and target angular velocity for the turn
            duration_multiplier = 0.0
            turn_angular_vel = 0.0
            # ASSUMPTION: Positive angular velocity -> Left turn (CCW)
            # ASSUMPTION: convert_twist_to_wheels handles kinematics correctly

            if angle_diff == 90: # Turn Right (CW)
                 rospy.loginfo("Turn Type: RIGHT 90 deg (CW)")
                 duration_multiplier = 1.0
                 turn_angular_vel = -abs(self.turn_speed_angular) # Negative for CW
            elif angle_diff == 270: # Turn Left (CCW)
                 rospy.loginfo("Turn Type: LEFT 90 deg (CCW)")
                 duration_multiplier = 1.0
                 turn_angular_vel = abs(self.turn_speed_angular) # Positive for CCW
            elif angle_diff == 180: # Turn Around (e.g., via Left/CCW)
                 rospy.loginfo("Turn Type: AROUND 180 deg (via CCW)")
                 duration_multiplier = 2.0 # Approx double time
                 turn_angular_vel = abs(self.turn_speed_angular) # Positive for CCW
            elif angle_diff == 0:
                 rospy.logwarn("Turn requested with 0 angle difference. Skipping turn.")
                 duration_multiplier = 0.0
                 turn_angular_vel = 0.0
            else: # Should not happen with cardinal directions
                 rospy.logerr(f"Unexpected angle diff {angle_diff} for turning. Stopping.")
                 self.current_state = State.ERROR
                 self.stop_robot()
                 return # Exit turning state logic

            self.effective_turn_duration = self.turn_duration_90_deg * duration_multiplier
            rospy.loginfo(f"Calculated turn duration: {self.effective_turn_duration:.2f}s")

            # Calculate wheel velocities for pure rotation (linear_vel = 0)
            left_vel, right_vel = self.convert_twist_to_wheels(0.0, turn_angular_vel)

            if self.effective_turn_duration > 0.01: # Only publish if turn is actually needed
                self.publish_wheel_velocity(left_vel, right_vel)
                rospy.loginfo(f"Publishing turn command: Ang={turn_angular_vel:.2f} -> L={left_vel:.2f}, R={right_vel:.2f}")
            else:
                # If no turn needed, force completion immediately
                self.turn_start_time = current_time - rospy.Duration(self.effective_turn_duration + 1.0)


        # Check if Turn is Complete
        elapsed_time = (current_time - self.turn_start_time).to_sec()

        if elapsed_time >= self.effective_turn_duration:
            rospy.loginfo(f"Turn towards {self.target_heading} complete (elapsed: {elapsed_time:.2f}s).")
            self.stop_robot()
            # Clear turn-specific variables
            self.target_heading = None
            self.pre_turn_heading = None
            self.turn_start_time = None
            self.effective_turn_duration = 0.0
            # Transition Back to Following
            rospy.loginfo(f"Transition: {self.current_state.name} -> {State.FOLLOWING.name}")
            self.current_state = State.FOLLOWING
            self.last_state_change = rospy.Time.now()
            # Optional short pause after turning before following again
            # rospy.sleep(0.2)
        # else: Turn is still in progress, wheels command was already sent on initiation

    def handle_error_state(self):
        """Logic for the ERROR state."""
        rospy.logerr_throttle(5.0, f"Robot in ERROR state. Stopping all movement.")
        self.stop_robot()
        # Consider adding recovery logic or manual intervention triggers here.


    def run(self):
        """The main control loop executing the state machine."""
        rospy.loginfo(f"[{rospy.get_name()}] Starting main loop...")
        while not rospy.is_shutdown():
            state_before_cycle = self.current_state

            # --- State Machine Execution ---
            if self.current_state == State.FOLLOWING:
                self.handle_following_state()
            elif self.current_state == State.STOPPED_AT_JUNCTION:
                self.handle_stopped_at_junction_state()
            elif self.current_state == State.TURNING:
                self.handle_turning_state()
            elif self.current_state == State.ERROR:
                self.handle_error_state()
            else:
                rospy.logfatal(f"Unknown state: {self.current_state}! Entering ERROR state.")
                self.current_state = State.ERROR
                self.stop_robot()

            # Log state transitions clearly
            if self.current_state != state_before_cycle:
                 rospy.loginfo(f"State changed: {state_before_cycle.name} -> {self.current_state.name}")

            try:
                self.rate.sleep()
            except rospy.ROSTimeMovedBackwardsException:
                 rospy.logwarn("ROS time moved backwards, ignoring sleep.")
            except rospy.ROSInterruptException:
                 rospy.loginfo("ROSInterruptException during sleep. Exiting run loop.")
                 break