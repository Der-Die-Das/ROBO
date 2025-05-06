# Build:
#  docker build --rm -t robo/ros-ur .

FROM ros:noetic
LABEL maintainer "Florian Herzog <florian.herzog@hslu.ch>"

SHELL ["/bin/bash","-c"]
ENV CATKIN_WS=/root/catkin_ws
RUN mkdir -p $CATKIN_WS/src

# install universal robot ros repository
RUN apt-get update && apt-get install -y git pip\
     && rm -rf /var/lib/apt/lists/*

# download robotiq gripper visualization and ur description

RUN cd ${CATKIN_WS}/src \
     && git clone -n --depth=1 --filter=tree:0 https://github.com/ros-industrial/robotiq \
     && cd robotiq \
     && git sparse-checkout set --no-cone robotiq_2f_140_gripper_visualization robotiq_2f_85_gripper_visualization \
     && git checkout

RUN cd ${CATKIN_WS}/src \
     && git clone -n --depth=1 --filter=tree:0 https://github.com/ros-industrial/universal_robot \
     && cd universal_robot \
     && git sparse-checkout set --no-cone ur_description \
     && git checkout

RUN cd ${CATKIN_WS}/src \
     && git clone https://gitlab.com/fherzog/ur_primary_client.git \
     && git clone https://gitlab.com/fherzog/ur-lab.git
  
# Install dependency
RUN pip install urdf_parser_py


# Build the catkin workspace
RUN cd ${CATKIN_WS} \
     && apt update -qq \
     && source /opt/ros/${ROS_DISTRO}/setup.bash \
     && rosdep update \
     && rosdep install --from-paths src --ignore-src -y \
     && catkin_make

RUN chmod +x ${CATKIN_WS}/src/ur_primary_client/src/client.py
RUN echo 'source /opt/ros/${ROS_DISTRO}/setup.bash' >> /root/.bashrc
RUN echo 'source /root/catkin_ws/devel/setup.bash' >> /root/.bashrc


