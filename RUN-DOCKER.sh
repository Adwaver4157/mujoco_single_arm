#! /bin/bash

docker run -it --rm --name $1_mujoco \
               --gpus all --shm-size=16gb \
               --mount type=bind,source="$(pwd)",target=/root/workspace \
               -p 5800:5900 -p 6006:6006 -p 8088:8888 \
               adwaver4157/mujoco:latest