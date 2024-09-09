FROM ubuntu:18.04
FROM python:3.6.9

RUN pip3 install --upgrade pip --index-url https://pypi.douban.com/simple

WORKDIR /workspace
ADD . /workspace

RUN apt-get update
# RUN echo "#!/bin/sh\nexit 0" > /usr/sbin/policy-rc.d
# RUN apt-get install --assume-yes apt-utils 
RUN apt install -y clang-6.0 graphviz protobuf-compiler libprotobuf-dev libboost-all-dev vim
RUN apt install -y python python-dev python-setuptools gcc libtinfo-dev zlib1g-dev build-essential libedit-dev libxml2-dev python-pip
RUN apt-get install -y autoconf automake libtool curl make g++ unzip
RUN pip install -r requirements.txt --index-url https://pypi.douban.com/simple
RUN pip install --upgrade protobuf

CMD ["/bin/bash"]
