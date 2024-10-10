#!/usr/bin/env bash

docker run -it \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --ipc=host \
    --shm-size 8G \
    -v /home/loctran/projects/extension-cpp-rocm:/workspace/extension-cpp-rocm \
    rocm/pytorch:latest
