FROM ubuntu:22.04
LABEL maintainer="Johann Fredrik Jadebeck <j.jadebeck@juelich.de>"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get clean -y
RUN apt-get update -y

# build tools
RUN apt-get install -y apt-utils
RUN apt-get install -y build-essential 
RUN apt-get install -y software-properties-common 
RUN apt-get install -y git
RUN apt-get install -y cmake 
RUN apt-get install -y clang
RUN apt-get install -y doxygen 
RUN apt-get install -y clang-tidy 
RUN apt-get install -y clang-format
RUN apt-get install -y cppcheck
RUN apt-get install -y python3-pip

# hops dependencies
RUN apt-get install -y libeigen3-dev 
RUN apt-get install -y libhdf5-dev 
RUN apt-get install -y libsbml5-dev 
RUN apt-get install -y libmpich-dev 
RUN apt-get install -y libbz2-dev
RUN apt-get install -y coinor-libclp-dev 
RUN apt-get install -y libboost-all-dev 
RUN apt-get install -y libtbb-dev

# convenience
RUN apt-get install -y vim 
RUN apt-get install -y ssh-client
RUN apt-get install -y xsltproc


# for creating bagdges
RUN python3 -m pip install anybadge
# for code quality
RUN python3 -m pip install cppcheck-codequality

RUN useradd -ms /bin/bash hops_user
USER hops_user

CMD tail -f /dev/null
