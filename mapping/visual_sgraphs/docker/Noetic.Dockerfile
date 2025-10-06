FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

# User and group setup
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Deletes user if already in container
RUN if id -u $USER_UID ; then userdel `id -un $USER_UID` ; fi

##### Environment variables #####
ENV CUDA_HOME=/usr/local/cuda
 
##### Essential packages #####
RUN apt-get update && apt-get install -y \
    python3-pip \
    python-is-python3 \
    git \
    openssh-client \
    wget \
    vim \
    curl \
    libeigen3-dev \
    build-essential
 
##### Install ROS 1 - Noetic #####
# Setup environment
ENV ROS_DISTRO=noetic
 
# Setup keys
RUN echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
 
# Install ros packages
RUN apt-get update && apt-get install -y \
    ros-$ROS_DISTRO-desktop-full
 
# ROS related packages
RUN apt-get install -y \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    python3-catkin-tools \
    ros-$ROS_DISTRO-pcl-ros \
    ros-$ROS_DISTRO-backward-ros \
    ros-$ROS_DISTRO-rviz-visual-tools \
    ros-$ROS_DISTRO-hector-trajectory-server \
    ros-$ROS_DISTRO-realsense2-camera \
    ros-$ROS_DISTRO-rgbd-launch
 
# ROS dependencies
RUN rosdep init
RUN rosdep update

# Create new user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
 
##### Python environment setup #####
# PyTorch and related libraries - networkx needs to be installed first because of version issue
RUN pip3 install networkx==3.1
RUN pip3 install torch==2.0.1+cu118 -f  https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
 
# Detectron2 and CLIP
# Compute Capability 7.5 for T600 (SnT laptop) and 7.0 for V100 (HPC - Iris)
ARG TORCH_CUDA_ARCH_LIST="7.5;7.0+PTX"
ENV FORCE_CUDA="1"
RUN pip3 install 'git+https://github.com/facebookresearch/detectron2.git'
RUN pip3 install 'git+https://github.com/openai/CLIP.git'
 
##### SSH keys for GitHub #####
 
# Define the SSH keys as build arguments for latter mounting
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts
 
##### Clone repositories #####
# Pangolin
RUN apt-get install libepoxy-dev -y
WORKDIR /opt/
RUN git clone --branch v0.8 --depth 1 https://github.com/stevenlovegrove/Pangolin.git && \
    cd Pangolin && \
    mkdir build && cd build && \
    cmake .. && \
    make -j && \
    make install
 
# Cmake
ARG version=3.22
ARG build=1
WORKDIR /tmp
RUN wget https://cmake.org/files/v$version/cmake-$version.$build.tar.gz
RUN tar -xzvf cmake-$version.$build.tar.gz
WORKDIR /tmp/cmake-$version.$build
RUN ./bootstrap
RUN make -j8
RUN make install
 
# ROS packages: visual sgraphs, semantic segmenter, aruco ros
RUN mkdir -p /workspace/src
WORKDIR /workspace/src/
 
# Mount the SSH keys for cloning private repositories
RUN --mount=type=ssh git clone git@github.com:snt-arg/visual_sgraphs.git
RUN --mount=type=ssh git clone -b ros1-noetic git@github.com:snt-arg/scene_segment_ros.git
RUN --mount=type=ssh git clone -b noetic-devel git@github.com:pal-robotics/aruco_ros.git

# Other libraries
WORKDIR /workspace/src/visual_sgraphs/docker
RUN pip3 install -r requirements.txt

WORKDIR /workspace/src/
 
# For ROS package: mav_voxblox_planning
RUN --mount=type=ssh git clone git@github.com:snt-arg/mav_voxblox_planning.git
RUN --mount=type=ssh wstool init . ./mav_voxblox_planning/install/install_ssh.rosinstall
RUN --mount=type=ssh wstool update
 
# Download the YOSO checkpoint
RUN wget https://github.com/hujiecpp/YOSO/releases/download/v0.1/yoso_res50_coco.pth
RUN mv yoso_res50_coco.pth /workspace/src/scene_segment_ros/include/
 
# Build the workspace
WORKDIR /workspace/
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin build -j10 -DCMAKE_BUILD_TYPE=Release && rosclean purge -y"
 
##### Miscalleanous #####
RUN ldconfig

# Copy the proper ArUco marker launch file
RUN cp /workspace/src/visual_sgraphs/doc/template_aruco_ros.launch /workspace/src/aruco_ros/aruco_ros/launch/marker_publisher.launch

# Copy the proper RealSense launch file and preset
RUN cp /workspace/src/visual_sgraphs/doc/RealSense/rs_d435_rgbd.launch /opt/ros/noetic/share/realsense2_camera/launch/rs_rgbd.launch
RUN cp /workspace/src/visual_sgraphs/doc/RealSense/presets/HighAccuracyPreset.json /opt/ros/noetic/share/realsense2_camera/launch/includes/HighAccuracyPreset.json
 
##### Clean up #####
# Remove the apt list files
RUN rm -rf /var/lib/apt/lists/*
 
# Remove packages no longer needed
RUN apt-get clean && apt-get autoremove -y
 
# Remove the ssh keys
RUN rm -rf /root/.ssh/
 
##### Build entrypoint #####
RUN echo "#!/bin/bash" >> /entrypoint.sh \
    && echo "echo \"source /opt/ros/$ROS_DISTRO/setup.bash\" >> ~/.bashrc" >> /entrypoint.sh \
    && echo "echo \"source /workspace/devel/setup.bash\" >> ~/.bashrc" >> /entrypoint.sh \
    && echo 'exec "$@"' >> /entrypoint.sh \
    && chmod a+x /entrypoint.sh
 
WORKDIR /workspace/
 
ENTRYPOINT ["/entrypoint.sh"]
USER $USERNAME
CMD ["/bin/bash"]
SHELL ["/bin/bash"]