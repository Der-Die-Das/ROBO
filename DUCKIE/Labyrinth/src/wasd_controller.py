#!/usr/bin/env python

import rospy
import sys, select, termios, tty # For keyboard input

# ---!!!! VERY IMPORTANT !!!! ---
# 1. VERIFY and CHANGE the import statement based on where WheelsCmdStamped is defined in YOUR project.
# 2. VERIFY and CHANGE the field names '.vel_left' and '.vel_right' if your message uses different names.
# Example: If it's in your own package 'my_robot_msgs':
# from my_robot_msgs.msg import WheelsCmdStamped
try:
    # Common location for Duckiebots, replace if necessary
    from duckietown_msgs.msg import WheelsCmdStamped
except ImportError:
    rospy.logfatal("Cannot import WheelsCmdStamped. Please install the necessary package"
                   " (e.g., duckietown_msgs) or adjust the import statement above.")
    sys.exit(1)
# ---!!!!!!!!!!!!!!!!!!!!!!!!!! ---


# --- Configuration ---
# <<<--- CHANGE THIS TO YOUR ROBOT'S NAME --->>>
ROBOT_NAME = "rho" # Or get from parameter server: rospy.get_param("~robot_name", "default")
# <<<---------------------------------------->>>

WHEEL_SPEED = 0.25  # Adjust velocity scale as needed (depends on robot/units)
TOPIC_NAME = f"/{ROBOT_NAME}/wheels_driver_node/wheels_cmd"

# --- Terminal Input Handling ---
# Store original terminal settings
try:
    settings = termios.tcgetattr(sys.stdin)
except termios.error:
    rospy.logerr("Could not get terminal attributes. Are you running in a valid terminal?")
    settings = None

def getKey():
    """Gets a single key press from the terminal."""
    if not settings:
        return None
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1) # Timeout 0.1 seconds
    key = None
    if rlist:
        key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

# --- Main Control Logic ---
def teleop():
    """Main function to run the WASD teleoperation."""
    if not settings:
        rospy.logerr("Terminal settings not available. Exiting.")
        return

    rospy.init_node('wasd_direct_wheels_teleop', anonymous=True)

    # Use the specific topic name and message type
    pub = rospy.Publisher(TOPIC_NAME, WheelsCmdStamped, queue_size=1)
    rospy.loginfo(f"Publishing wheel commands to: {TOPIC_NAME}")
    rospy.loginfo(f"Using Message Type: {WheelsCmdStamped._type}")
    rospy.loginfo("Please verify the import and field names ('vel_left', 'vel_right') if errors occur.")


    print("WASD Direct Wheel Teleop Initialized.")
    print("---------------------------")
    print("Moving around:")
    print("   w")
    print("a  s  d")
    print("\n(Press any other key or wait to stop)")
    print("CTRL-C to quit")
    print("---------------------------")

    target_vel_left = 0.0
    target_vel_right = 0.0

    try:
        while not rospy.is_shutdown():
            key = getKey()
            if key is None and settings:
                # Stop on timeout/no key
                target_vel_left = 0.0
                target_vel_right = 0.0

            elif key == 'w': # Forward
                target_vel_left = WHEEL_SPEED
                target_vel_right = WHEEL_SPEED
            elif key == 's': # Backward
                target_vel_left = -WHEEL_SPEED
                target_vel_right = -WHEEL_SPEED
            elif key == 'a': # Turn Left
                target_vel_left = -WHEEL_SPEED*0.8 # Slower inside wheel slightly
                target_vel_right = WHEEL_SPEED*0.8
            elif key == 'd': # Turn Right
                target_vel_left = WHEEL_SPEED*0.8
                target_vel_right = -WHEEL_SPEED*0.8 # Slower inside wheel slightly
            elif key == '\x03': # CTRL-C
                 break
            else: # Any other key stops the robot
                target_vel_left = 0.0
                target_vel_right = 0.0

            print(target_vel_left)
            # Create and publish the WheelsCmdStamped message
            msg = WheelsCmdStamped()
            msg.header.stamp = rospy.Time.now()
            # ---!!! Verify these field names !!! ---
            msg.vel_left = float(target_vel_left)
            msg.vel_right = float(target_vel_right)
            # ---!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ---

            pub.publish(msg)

    except Exception as e:
        rospy.logerr(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Always try to stop the robot and restore terminal settings on exit
        rospy.loginfo("Stopping robot and restoring terminal...")
        msg = WheelsCmdStamped() # Zero velocity command
        msg.header.stamp = rospy.Time.now()
        # ---!!! Verify these field names !!! ---
        msg.vel_left = 0.0
        msg.vel_right = 0.0
        # ---!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ---

        if 'pub' in locals() and pub:
             try:
                 pub.publish(msg)
                 rospy.sleep(0.1) # Give publisher a moment
                 pub.publish(msg) # Publish again just in case
             except Exception as e:
                  rospy.logerr(f"Error publishing stop message: {e}")


        if settings:
             termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        rospy.loginfo("WASD Direct Wheel Teleop finished.")

if __name__ == "__main__":
    teleop()