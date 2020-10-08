# 从源中pull
# tensorflow目前不支持10.1以上版本
FROM ubuntu:18.04

RUN rm -f /etc/apt/sources.list && rm -rf /var/lib/apt/lists/* &&\
echo "deb http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse\n\
deb http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse\n\
deb http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse\n\
deb http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse\n\
deb http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse\n\
deb-src http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse\n\
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse\n\
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse\n\
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse\n\
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse"\
    >>/etc/apt/sources.list
# 安装
RUN apt-get update &&\
apt-get install -y \
    software-properties-common \
    curl \
    apt-utils \
    python3 \
    python3-dev \
    git \
    && apt-get clean 

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py && \
    python3 -m pip config set global.index-url http://pypi.douban.com/simple/ &&\
    python3 -m pip config set global.trusted-host pypi.douban.com

RUN python3 -m pip --no-cache-dir install \
    numpy==1.16.4 \
    scipy \
    scikit-learn \
    opencv-python \
    Pillow \
    matplotlib \
    Cython \
    numba

EXPOSE 1000-1010

CMD ["/bin/bash"]

# 图形显示
# docker run -it --gpus all --net=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --name 'lyd' -v ~/docker_v:/data yolo_v4 bash