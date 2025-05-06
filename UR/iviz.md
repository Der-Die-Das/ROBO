# IVIZ 

This applications allows to visualize robots and certain ROS topics on mobile devices. The source code is open-source and can be found [here](https://github.com/KIT-ISAS/iviz).

You have two options to use IVIZ: use a tablet provided by your teacher or install it on an android mobile device.

## Installation on Android Device

1. Download the [.apk](https://github.com/KIT-ISAS/iviz/blob/devel/iviz/Binaries/android-7.11.21.apk) file from the devel branch of the git repo.
1. Install the application on your device. Make sure that your device allows installation of apks from the given source.


## Configuration
1. With your mobile device, connect to the network `robo-ur` and open IVIZ.
3. In IViz, click on `Connection` and set

    |  |  | |
    |---|---|---|
    |**Master URI** |`http://192.168.1.110:11311` | IP of the NUC corresponding to your Universal Robot (see table below) and port 11311
    | **My URI** | `http://192.168.1.46:5678/` | IP and random port of your mobile device running Iviz.


    The NUCs for the UR Robots have the following IP addresses:

    || |
    |-|-|
    | UR3e on the left | 192.168.1.110 
    | UR3e on the right  | 192.168.1.111
   

    To get the IP of your device, have a look at this [website](https://help.simpletelly.com/article/329-how-to-find-your-android-device-ip-address).


4. Click on connect.

5. Go back to the [Readme.md](Readme.md) and follow the steps until you have launched the `virtual_robot.launch` or the `real_robot.launch`.

5. In Iviz, click on `Module` and add `Robot`. Inside the new window, select `/robot_description` for the field `Load from Source Parameter`.

6. The Universal Robot should appear on your screen.