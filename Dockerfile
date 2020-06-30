FROM ubuntu:20.04
LABEL Maintainer="Johann Fredrik Jadebeck <johann.fredrik@jadebeck.dev>"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y
RUN apt-get install -y apt-utils
RUN apt-get install -y build-essential software-properties-common cmake libeigen3-dev \
    liblpsolve55-dev lp-solve libxerces-c-dev libhdf5-dev doxygen libncurses5-dev libncursesw5-dev \
    libsbml5-dev mpich libmpich-dev git
RUN apt-get install -y bzip2 libbz2-dev
RUN apt-get install -y coinor-clp coinor-libclp-dev

WORKDIR /home

ADD benchmarks /home/benchmarks
ADD cmake /home/cmake
ADD docs /home/docs
ADD examples /home/examples
ADD include /home/include
ADD resources /home/resources
ADD src /home/src
ADD tests /home/tests
ADD CMakeLists.txt /home/CMakeLists.txt

RUN mkdir cmake-build-debug
RUN mkdir cmake-build-release

WORKDIR /home/cmake-build-debug
RUN cmake .. -DCMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu
RUN make -j4
RUN make test ARGS=j4

WORKDIR /home/cmake-build-release
RUN cmake .. -DCMAKE_BUILD_TYPE=Release
RUN make -j4
RUN make test ARGS=j4
