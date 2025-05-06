# Universal Robot UR3e Lab

## Requirements
- Laptop with VS Code and the Docker and Dev Container extensions
- Tablet (ask your teacher) or android phone with IVIZ (see [Installation](iviz.md))

To visualize the virtual robot, you need either access to the mini PC (NUC) connected to the `robo-ur` network or use a Linux machine either physical or virtual.


## Virtual Robot using the NUC

1. Connect your laptop to the `robo-ur` network (password: `denavithartenberg`).

1. Make sure the corresponding NUC is running.

1. In VS Code, change the `DOCKER_HOST`variable

   - **Windows**: navigate to *File* > *Preferences* > *Settings* > *Docker Environment*
   - **Mac**: click on *Manage* icon > *Settings* > *Docker* > *Docker: Environment*
   
    and add the entry

    `DOCKER_HOST`   `ssh://robo@192.168.1.110` for the left UR Robot, or 
   
   `DOCKER_HOST`    `ssh://robo@192.168.1.111` for the right robot.

1. You may want to set up a ssh key pair to connect to the NUC. The password for the NUC is `denavithartenberg`. In a command shell, enter

   - **Windows**: 
   
         type %USERPROFILE%\.ssh\id_rsa.pub | ssh robo@192.168.1.110 "cat >> .ssh/authorized_keys"` 

   - **Mac**: 
        
         ssh-copy-id robo@192.168.1.110

   using the right IP address.



1. Inside the [docker-compose.yml](docker-compose.yml) set the correct IP adress for **both** services:

        ROS_MASTER_URI=http://192.168.1.110:11311

   or

        ROS_MASTER_URI=http://192.168.1.111:11311


1. Start the docker containers inside VS Code with a right-click on `docker-compose.yml` and `Compose Up`. 

1. In VS Code, on the lower left: click on `open a remote window` and select `attach to running container` and select the `ur-lab-ros-1` container.

1. Inside the container, start the launch file with the terminal

        roslaunch ur_lab virtual_robot.launch
    

1. Develop your code in `ur-lab/src/kinematics.py` Whenever you change something in the code, you can simply relaunch the ROS node. You don't need to build the container or the ROS catkin workspace again.
    
1. If you change your code, stop the ROS node (Ctrl+c / Ctrl+z) and relaunch it with:

        roslaunch ur_lab virtual_robot.launch

   or start the `kinematics.py` in a separate terminal window
        
        rosrun ur_lab kinematics.py

## Two Virtual Robots

1. Start the two robots:

        roslaunch ur_lab two_virtual_robots.launch

   This starts an `alpha` and a `beta` robot.

1. Work either in the `kinematics_with_two_robots_alpha.py` or the `kinematics_with_two_robots_beta.py`.

1. Each time you start the two robots with the `roslaunch` command, both python files are executed. In order not to interfe with the other group, comment out the line in `ur-lab/launch/two_virtual_robots.launch` that starts the other node:

        <!--node name="kinemtics_alpha" pkg="ur_lab" type="kinematics_with_two_robots_alpha.py" /-->


## Virtual Robot using Linux

1. Follow the instructions above starting at step 5.
2. In IVIZ, use the IP of the machine running the docker containers instead of the NUC's IPs.

## Real Robot


It is **mandatory** to get an instruction from your teacher before manipulating the robots!

1. Connect to the wireless network `robo-ur` with the password `denavithartenberg`.

1. Start the mini PC (NUC) corresponding to your robot.

1. Start the Universal Robot with the teach pendant. 

1. Activate the Universal Robot and set it to *Remote Control* (*Fernsteuerung*).

1. Start the docker containers and connect to the `ur-lab-ros-1` container with VS Code.
        
1. Lauch ROS with

        roslaunch ur_lab real_robot.launch

1. To send commands to the robot, you can use the `move_real_robot.py` as a template.

        rosrun ur_lab move_real_robot.py


