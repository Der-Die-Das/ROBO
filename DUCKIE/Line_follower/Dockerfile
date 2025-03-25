FROM duckietown/dt-ros-commons:daffy

RUN cd /code/catkin_ws/src \
    && git clone https://gitlab.com/fherzog/duckie-lab.git

RUN cd /code/catkin_ws \
    && catkin build 

RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
    echo "source /code/catkin_ws/devel/setup.bash" >> ~/.bashrc 

COPY src/ /code/catkin_ws/src