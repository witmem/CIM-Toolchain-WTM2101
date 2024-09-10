FROM python:3.6.9 as build-stage
SHELL ["bash", "-c"]

WORKDIR /tmp

RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y wget
RUN apt-get install -y tar
RUN apt-get install -y cmake
RUN apt-get install -y llvm-6.0
RUN apt-get install -y llvm-6.0-dev

# install cmake
COPY ./tools/cmake-3.18.6.tar.gz /tmp/
RUN tar xvf ./cmake-3.18.6.tar.gz

WORKDIR /tmp/cmake-3.18.6/build
RUN cmake ..
RUN make -j 8
RUN make install

# install bgl
WORKDIR /tmp/
RUN wget https://newcontinuum.dl.sourceforge.net/project/boost/boost/1.83.0/boost_1_83_0.tar.bz2
RUN tar xvf boost_1_83_0.tar.bz2

WORKDIR /tmp/boost_1_83_0/
RUN ./bootstrap.sh
RUN ./b2 install

WORKDIR /tmp/
RUN rm boost_1_83_0.tar.bz2
RUN rm -rf boost_1_83_0

WORKDIR /tmp
COPY ./witin_mapper /tmp/witin_mapper


# release build
FROM python:3.6.9

USER root

WORKDIR /workspace

RUN apt-get update
RUN apt-get -y install cmake
RUN apt-get install -y llvm-6.0
RUN apt-get install -y llvm-6.0-dev
RUN apt-get install -y vim

RUN pip install numpy==1.19.5
RUN pip install protobuf==3.19.6
RUN pip install onnx==1.10.2
RUN pip install decorator==5.1.0
RUN pip install typed-ast==1.5.1
RUN pip install Pillow==8.4.0
RUN pip install matplotlib==3.3.4
RUN pip install scipy==1.5.4
RUN pip install attrs==21.4.0
RUN pip install torch==1.10.1
RUN pip install pytest==6.2.5

COPY --from=build-stage /tmp/witin_mapper ./witin_mapper

# install fonts
COPY ./tools/Gargi.ttf /usr/share/fonts/Gargi.ttf
RUN fc-cache -fv

WORKDIR /tmp/

# install bgl
WORKDIR /tmp/
RUN wget https://newcontinuum.dl.sourceforge.net/project/boost/boost/1.83.0/boost_1_83_0.tar.bz2
RUN tar xvf boost_1_83_0.tar.bz2

WORKDIR /tmp/boost_1_83_0/
RUN ./bootstrap.sh
RUN ./b2 install

WORKDIR /tmp/
RUN rm boost_1_83_0.tar.bz2
RUN rm -rf boost_1_83_0

# install cmake
WORKDIR /tmp/
COPY ./tools/cmake-3.18.6.tar.gz /tmp/
RUN tar xvf ./cmake-3.18.6.tar.gz

WORKDIR /tmp/cmake-3.18.6/build/
RUN cmake ..
RUN make -j 8
RUN make install

WORKDIR /tmp/
RUN rm -rf cmake-3.18.6.tar.gz cmake-3.18.6

# install protobuf
WORKDIR /tmp/
RUN git clone -b v3.7.1 --depth=1 --recursive https://github.com/protocolbuffers/protobuf.git

WORKDIR /tmp/protobuf
RUN bash ./autogen.sh
RUN bash ./configure
RUN make -j8
RUN make check
RUN make install

WORKDIR /tmp/
RUN rm -rf protobuf

# install gtest
WORKDIR /tmp/
RUN git clone --depth=1 --recursive https://github.com/google/googletest.git

WORKDIR /tmp/googletest/build/
RUN cmake ..
RUN make -j8
RUN make install

WORKDIR /tmp/
RUN rm -rf googletest

# install jsoncpp
WORKDIR /tmp/
RUN git clone -b 1.9.3 --depth=1 --recursive https://github.com/open-source-parsers/jsoncpp.git

WORKDIR /tmp/jsoncpp/build/
RUN cmake ..
RUN make -j8
RUN make install

WORKDIR /tmp/
RUN rm -rf jsoncpp

ENV LD_LIBRARY_PATH=/workspace/witin_mapper/build/lib:/workspace/witin_mapper/build:/usr/local/lib
ENV PYTHONPATH=/workspace/witin_mapper/python
ENV TVM_HOME=/workspace/witin_mapper

# clean up
RUN rm /var/lib/apt/lists/* -rf

WORKDIR /workspace


