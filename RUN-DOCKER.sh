#! /bin/bash

if [ "$2" == "ssh" ]; then
    echo "SSH and VNC mode"
    docker run -it --rm --name $1_mujoco \
                   --gpus all --shm-size=16gb \
                   --mount type=bind,source="$(pwd)",target=/root/workspace \
                   -p 5800:5900 -p 6006:6006 -p 8088:8888 \
                   adwaver4157/mujoco:latest
else
    echo "Local mode"
    xhost +local:
    docker run -it --rm --name $1_mujoco \
                   --gpus all --shm-size=16gb \
                   --env DISPLAY=${DISPLAY} \
                   --net host \
                   -v /tmp/.X11-unix:/tmp/.X11-unix \
                   --mount type=bind,source="$(pwd)",target=/root/workspace \
                   adwaver4157/mujoco:latest
fi