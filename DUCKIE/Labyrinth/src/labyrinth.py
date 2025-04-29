#!/usr/bin/env python

import rospy
import traceback
from robot_controller import RobotController

def main():
    rospy.init_node("maze_solver_node")
    controller = None # Define outside try for finally clause
    try:
        controller = RobotController()
        controller.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROSInterruptException caught in main. Shutting down.")
    except Exception as e:
        rospy.logfatal(f"Unhandled exception in main execution: {e}")
        traceback.print_exc() # Print stack trace for debugging
    finally:
         # Ensure robot stops even if RobotController initialization failed
         # or run() exited unexpectedly. The controller's shutdown hook is
         # primary, but this adds another layer of safety.
         if controller is not None:
             rospy.loginfo("Main scope finally block: ensuring robot is stopped.")
             # Use the controller's stop method if available
             controller.stop_robot()
             rospy.sleep(0.5) # Give it time
             controller.stop_robot()
         else:
              # If controller failed to init, we might not have a publisher easily accessible.
              # Relying on the shutdown hook is crucial here if it got registered.
              rospy.logwarn("Controller object was not initialized. Cannot guarantee stop from main finally block.")
         rospy.loginfo(f"{rospy.get_name()} finished.")


if __name__ == "__main__":
    main()