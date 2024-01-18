FROM ubuntu:20.04

ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update &&  \
    apt-get install -y software-properties-common libssl-dev wget git ninja-build build-essential  \
      curl zip unzip tar bison pkg-config python3 python3-pip libxrandr-dev libxcursor-dev  \
      libxcomposite-dev libxdamage-dev libxtst-dev libdbus-1-dev libgl-dev libxi-dev libxtst-dev && \
    git clone https://github.com/Microsoft/vcpkg.git &&  \
    ./vcpkg/bootstrap-vcpkg.sh -disableMetrics &&  \
    chmod +x /vcpkg/vcpkg &&  \
    ln -sf /vcpkg/vcpkg /bin/vcpkg &&  \
    cd / &&  \
    pip3 install disutils && \
    wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 14 && \
    wget https://github.com/Kitware/CMake/releases/download/v3.28.1/cmake-3.28.1.tar.gz && \
    tar -zxvf cmake-3.28.1.tar.gz && \
    cd cmake-3.28.1 && \
    ./bootstrap && \
    make -j10 && make install
