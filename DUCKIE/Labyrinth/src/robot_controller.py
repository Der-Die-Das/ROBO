# src/maze_solver_pkg/robot_controller.py
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped
import time # Using Python time for simple turn duration

# Import local modules (assuming they are in the same Python package)
from state import State # Assuming State enum includes DRIVING_INTO_JUNCTION
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
        self.fwd_speed_following = rospy.get_param("~fwd_speed_following", 0.08)
        kp_line_follow = rospy.get_param("~kp_line_follow", 6.0) # Gain for simple follower
        # Turning Params
        self.turn_speed_angular = rospy.get_param("~turn_speed_angular", 1) # rad/s
        self.turn_duration_90_deg = rospy.get_param("~turn_duration_90_deg", 2) # Seconds (NEEDS TUNING)
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
        # *** NEW PARAMETER ***
        self.drive_into_junction_duration = rospy.get_param("~drive_into_junction_duration", 1.5) # Seconds to drive forward after detection
        self.drive_into_junction_speed = rospy.get_param("~drive_into_junction_speed", 0.08) # Speed during the short forward push


        # --- State Machine ---
        self.current_state = State.FOLLOWING
        self.target_heading = None # Store the heading we want to turn towards
        self.pre_turn_heading = None # Store the heading *before* the turn decision
        self.turn_start_time = None
        self.effective_turn_duration = 0.0 # Store calculated duration for the current turn
        self.drive_forward_start_time = None # Timer for the short drive into junction
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
        self.image_width = rospy.get_param("~camera_image_width", 640) # Default width
        self.line_follower = SimpleLineFollower(kp=kp_line_follow, img_width=self.image_width)

        self.detector = TapeMazeDetector(
            current_heading="north", # Initial assumption
            border_crop_pixels=detector_border_crop,
            angle_tolerance=detector_angle_tol,
            forward_top_threshold_px=detector_forward_thresh,
            mask_crop_percentage=detector_mask_crop,
            mask_lower_threshold=detector_mask_thresh,
        )
        self.solver = TremauxSolver(threshold=self.tremaux_junction_threshold)

        # --- Data ---
        self.current_frame = None
        self.last_image_time = rospy.Time(0)

        rospy.loginfo(f"[{rospy.get_name()}] Initialized. State: {self.current_state.name}")
        rospy.loginfo(f"[{rospy.get_name()}] Following Speed: {self.fwd_speed_following:.2f}, Drive-In Speed: {self.drive_into_junction_speed:.2f} ({self.drive_into_junction_duration:.2f}s)")
        rospy.loginfo(f"[{rospy.get_name()}] Turn Speed: {self.turn_speed_angular:.2f} rad/s, Turn Duration (90deg): {self.turn_duration_90_deg:.2f}s")
        rospy.loginfo(f"[{rospy.get_name()}] Solver Dist Assumption: {self.segment_distance_assumption:.2f}m, Junction Threshold: {self.tremaux_junction_threshold:.2f}m")
        rospy.loginfo(f"[{rospy.get_name()}] Detector Initial Heading: {self.detector.current_heading}")

    def image_callback(self, msg):
        """Handles incoming compressed image messages."""
        try:
            self.current_frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            self.last_image_time = msg.header.stamp
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
        # Ensure the robot is stopped before exiting
        self.publish_wheel_velocity(0.0, 0.0)
        rospy.sleep(0.2) # Give publisher time
        self.publish_wheel_velocity(0.0, 0.0) # Send stop again just in case


    # --- State Handlers ---

    def handle_following_state(self):
        """Logic for the FOLLOWING state: Detect junctions and follow line."""
        if self.current_frame is None:
            self.stop_robot() # Stop if no image
            rospy.logwarn_throttle(2.0, "FOLLOWING: No image frame available. Stopping.")
            return

        # 1. Process Image for Junction Detection
        junction_detected = False
        available_dirs = [] # Store locally for this cycle
        try:
            junction_detected, available_dirs = self.detector.process_image(self.current_frame)
        except Exception as e:
             rospy.logerr(f"Error during tape detection: {e}")
             self.stop_robot()
             self.current_state = State.ERROR
             return

        # 2. Check for State Transition to DRIVING_INTO_JUNCTION
        time_since_last_change = (rospy.Time.now() - self.last_state_change).to_sec()
        # Add a small delay (e.g., > 1.0 sec) to prevent immediate re-detection after a turn
        if junction_detected and time_since_last_change > 1.0:
            rospy.loginfo(f"Detector (H: {self.detector.current_heading}): Junc={junction_detected}, Dirs={available_dirs}")
            rospy.loginfo(f"Transition: {self.current_state.name} -> {State.DRIVING_INTO_JUNCTION.name}")
            self.current_state = State.DRIVING_INTO_JUNCTION
            self.last_state_change = rospy.Time.now()
            self.drive_forward_start_time = None # Reset timer for the new state
            # *** CRITICAL: Store detected directions for the next state ***
            self.junction_available_dirs = available_dirs # Store for use in STOPPED state
            # *** DO NOT STOP HERE *** - Continue to the next state which handles the short drive
            return # Exit handler for this cycle

        # 3. Execute Action: Follow Line using SimpleLineFollower
        angular_vel_cmd, line_detected = self.line_follower.calculate_angular_velocity(self.current_frame)

        if line_detected:
            left_vel, right_vel = self.convert_twist_to_wheels(self.fwd_speed_following, angular_vel_cmd)
            self.publish_wheel_velocity(left_vel, right_vel)
        else:
            rospy.logwarn_throttle(1.0,"Line lost during FOLLOWING state. Stopping.")
            self.stop_robot()
            # Optional: Add recovery or transition to ERROR

    def handle_driving_into_junction_state(self):
        """Logic for DRIVING_INTO_JUNCTION: Drive straight for a fixed duration."""
        current_time = rospy.Time.now()

        # 1. Initialize Timer and Start Driving Forward
        if self.drive_forward_start_time is None:
            rospy.loginfo(f"Starting short drive into junction for {self.drive_into_junction_duration:.2f}s...")
            self.drive_forward_start_time = current_time
            # Command wheels to move straight at the specified speed
            left_vel, right_vel = self.convert_twist_to_wheels(self.drive_into_junction_speed, 0.0)
            self.publish_wheel_velocity(left_vel, right_vel)
            rospy.loginfo(f"Publishing drive-in command: Lin={self.drive_into_junction_speed:.2f} -> L={left_vel:.2f}, R={right_vel:.2f}")

        # 2. Continue Driving Forward (re-publish command)
        # Re-publish to ensure command persistence
        left_vel, right_vel = self.convert_twist_to_wheels(self.drive_into_junction_speed, 0.0)
        self.publish_wheel_velocity(left_vel, right_vel)

        # 3. Check if Duration Elapsed
        elapsed_time = (current_time - self.drive_forward_start_time).to_sec()

        if elapsed_time >= self.drive_into_junction_duration:
            rospy.loginfo(f"Short drive into junction complete (elapsed: {elapsed_time:.2f}s). Stopping.")
            self.stop_robot()
            # Reset timer variable
            self.drive_forward_start_time = None
            # Transition to STOPPED_AT_JUNCTION
            rospy.loginfo(f"Transition: {self.current_state.name} -> {State.STOPPED_AT_JUNCTION.name}")
            self.current_state = State.STOPPED_AT_JUNCTION
            self.last_state_change = rospy.Time.now()
            # Give a tiny pause after stopping before processing junction
            # rospy.sleep(0.1) # Optional small delay
        # else: Continue driving in the next cycle

    def handle_stopped_at_junction_state(self):
        """Logic for STOPPED_AT_JUNCTION: Call solver, decide next state."""
        # Now the robot is stopped slightly further into the junction
        rospy.loginfo_once("Entered STOPPED_AT_JUNCTION state (after short drive).")

        # Use the directions detected just before entering DRIVING_INTO_JUNCTION
        # We stored them in self.junction_available_dirs
        available_dirs = getattr(self, 'junction_available_dirs', []) # Use stored or empty list

        # Optional: Re-process image to *confirm* directions, but base decision on initially detected ones
        # if self.current_frame:
        #     _, confirmed_dirs = self.detector.process_image(self.current_frame)
        #     rospy.loginfo(f"STOPPED: Initially saw {available_dirs}, now confirming {confirmed_dirs}")
        # else:
        #     rospy.logwarn("STOPPED_AT_JUNCTION: No image frame available to confirm directions.")


        # 1. Make Decision using Solver
        if not available_dirs:
            rospy.logwarn("STOPPED_AT_JUNCTION: No available directions remembered from detection. Treating as dead end.")
            # Tremaux logic for dead end: Turn around (180 degrees)
            came_from = self.solver.opposites.get(self.solver.last_direction) if self.solver.last_direction else None

            if came_from:
                 rospy.logwarn(f"Planning 180 turn back towards {came_from}.")
                 self.pre_turn_heading = self.detector.current_heading
                 self.target_heading = came_from # Target is where we entered from
                 self.detector.update_heading(self.target_heading) # Update heading for after turn
                 rospy.loginfo(f"Detector heading updated to: {self.detector.current_heading}")
                 self.current_state = State.TURNING
                 self.last_state_change = rospy.Time.now()
                 self.turn_start_time = None # Reset turn timer flag
            else:
                 rospy.logerr("STOPPED_AT_JUNCTION: Dead end detected, but cannot determine entry direction! Entering ERROR state.")
                 self.current_state = State.ERROR
            return

        # Valid directions available, call solver
        try:
            chosen_direction = self.solver.junction_reached(self.segment_distance_assumption, available_dirs)
            rospy.loginfo(f"Solver recommends: {chosen_direction} (Current heading: {self.detector.current_heading}, Available: {available_dirs})")
        except Exception as e:
            rospy.logerr(f"Error calling Tremaux solver: {e}")
            self.current_state = State.ERROR
            return

        # 2. Decide Next Action Based on Solver Choice
        if chosen_direction == self.detector.current_heading:
             rospy.loginfo("Chosen direction is current heading. Proceeding forward.")
             rospy.loginfo(f"Transition: {self.current_state.name} -> {State.FOLLOWING.name}")
             self.current_state = State.FOLLOWING
             self.last_state_change = rospy.Time.now()
        else:
             # Prepare for turning
             self.pre_turn_heading = self.detector.current_heading
             self.target_heading = chosen_direction
             rospy.loginfo(f"Transition: {self.current_state.name} -> {State.TURNING.name} (From: {self.pre_turn_heading}, Target: {self.target_heading})")
             self.current_state = State.TURNING
             self.last_state_change = rospy.Time.now()
             self.turn_start_time = None
             self.detector.update_heading(self.target_heading) # Update heading for after turn
             rospy.loginfo(f"Detector heading updated to: {self.detector.current_heading} (for next segment)")

        # Clear the stored directions after use
        if hasattr(self, 'junction_available_dirs'):
            del self.junction_available_dirs


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

            current_angle = self.detector.cardinal_map.get(self.pre_turn_heading, 0)
            target_angle = self.detector.cardinal_map.get(self.target_heading, 0)
            angle_diff = (target_angle - current_angle + 360) % 360
            rospy.loginfo(f"Required turn angle: {angle_diff} degrees")

            duration_multiplier = 0.0
            turn_angular_vel = 0.0

            if angle_diff == 90: # Turn Right (CW)
                 rospy.loginfo("Turn Type: RIGHT 90 deg (CW)")
                 duration_multiplier = 1.0
                 turn_angular_vel = -abs(self.turn_speed_angular)
            elif angle_diff == 270: # Turn Left (CCW)
                 rospy.loginfo("Turn Type: LEFT 90 deg (CCW)")
                 duration_multiplier = 1.0
                 turn_angular_vel = abs(self.turn_speed_angular)
            elif angle_diff == 180: # Turn Around (e.g., via Left/CCW)
                 rospy.loginfo("Turn Type: AROUND 180 deg (via CCW)")
                 # Need to tune this multiplier based on observation
                 duration_multiplier = 1.9 # Might not be exactly 2.0 * 90deg duration
                 turn_angular_vel = abs(self.turn_speed_angular)
            elif angle_diff == 0:
                 rospy.logwarn("Turn requested with 0 angle difference. Skipping turn.")
                 duration_multiplier = 0.0
                 turn_angular_vel = 0.0
            else:
                 rospy.logerr(f"Unexpected angle diff {angle_diff} for turning. Stopping.")
                 self.current_state = State.ERROR
                 self.stop_robot()
                 return

            self.effective_turn_duration = self.turn_duration_90_deg * duration_multiplier
            rospy.loginfo(f"Calculated turn duration: {self.effective_turn_duration:.2f}s")

            left_vel, right_vel = self.convert_twist_to_wheels(0.0, turn_angular_vel)

            if self.effective_turn_duration > 0.01:
                self.publish_wheel_velocity(left_vel, right_vel)
                rospy.loginfo(f"Publishing turn command: Ang={turn_angular_vel:.2f} -> L={left_vel:.2f}, R={right_vel:.2f}")
            else:
                # Force completion immediately if no turn needed
                self.turn_start_time = current_time - rospy.Duration(self.effective_turn_duration + 0.1) # Ensure completion


        # Check if Turn is Complete
        elapsed_time = (current_time - self.turn_start_time).to_sec()

        if elapsed_time >= self.effective_turn_duration:
            rospy.loginfo(f"Turn towards {self.target_heading} complete (elapsed: {elapsed_time:.2f}s).")
            self.stop_robot()
            # Short pause after stopping turn before following
            rospy.sleep(0.2) # Give dynamics time to settle
            self.stop_robot() # Send stop again after pause

            # Clear turn-specific variables
            self.target_heading = None
            self.pre_turn_heading = None
            self.turn_start_time = None
            self.effective_turn_duration = 0.0
            # Transition Back to Following
            rospy.loginfo(f"Transition: {self.current_state.name} -> {State.FOLLOWING.name}")
            self.current_state = State.FOLLOWING
            self.last_state_change = rospy.Time.now()

        # else: Turn is still in progress, wheels command was already sent

    def handle_error_state(self):
        """Logic for the ERROR state."""
        rospy.logerr_throttle(5.0, f"Robot in ERROR state. Stopping all movement.")
        self.stop_robot()


    def run(self):
        """The main control loop executing the state machine."""
        rospy.loginfo(f"[{rospy.get_name()}] Starting main loop...")
        while not rospy.is_shutdown():
            state_before_cycle = self.current_state

            # --- State Machine Execution ---
            if self.current_state == State.FOLLOWING:
                self.handle_following_state()
            elif self.current_state == State.DRIVING_INTO_JUNCTION: # *** ADDED STATE ***
                self.handle_driving_into_junction_state()
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
        
        # Ensure robot is stopped cleanly on exit
        self.shutdown_hook()