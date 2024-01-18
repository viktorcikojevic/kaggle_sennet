#!/usr/bin/bash


rm -rf deploy_build/*
docker run -it --rm \
  -v /home/clay/research/kaggle/sennet/cpp:/cpp \
  sennet_build_image:20.04 bash

rm -rf /cpp/deploy_build && \
  mkdir -p /cpp/deploy_build && \
  cd /cpp/deploy_build && \
  cmake -G Ninja -DCMAKE_C_COMPILER=clang-14 -DCMAKE_CXX_COMPILER=clang++-14 -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_TOOLCHAIN_FILE=/vcpkg/scripts/buildsystems/vcpkg.cmake .. && \
  cmake --build . -j 10
