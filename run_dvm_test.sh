#!/bin/bash
xhost +
PATH_NAME="$(pwd -P)"
PATH_NAME+=":/code/DVM"

nvidia-docker container run \
	--runtime=nvidia		\
	-it \
	--rm						\
	-e GRANT_SUDO=yes --user root \
	-p 55530:55530 \
	-e DISPLAY=$DISPLAY \
	--gpus 'all'	\
	--volume="/tmp/.X11-unix:/tmp/.X11-unix" \
	--volume=$PATH_NAME \
	--name dvm_infer	\
	min00001/cuglmink:latest \
	/usr/bin/python code/DVM/simple_inference.py
