FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu18.04

RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y python3 python3-pip python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential libedit-dev libxml2-dev git
ADD .deps/software/* /opt/
ENV PATH $PATH:/opt/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04/bin/:/opt/cmake-3.25.0-rc1-linux-x86_64/bin:$PATH
RUN mkdir /opt/apache-tvm-src-v0.13.0/build/
COPY .deps/config.cmake /opt/apache-tvm-src-v0.13.0/build/
WORKDIR /opt/apache-tvm-src-v0.13.0/build
RUN cmake .. && make -j16
ENV TVM_HOME /opt/apache-tvm-src-v0.13.0
ENV PYTHONPATH $TVM_HOME/python:${PYTHONPATH}
RUN pip3 install --upgrade pip -i https://mirrors.ustc.edu.cn/pypi/web/simple && \
    pip3 install pytest numpy decorator pandas decorator attrs typing-extensions scipy psutil \
    tornado 'xgboost>=1.1.0' cloudpickle tensorflow-gpu==1.15 torch torchvision torchaudio\
    -i https://mirrors.ustc.edu.cn/pypi/web/simple