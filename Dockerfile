FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

SHELL ["/bin/bash", "-c"]
RUN apt-get update -qq && apt-get upgrade -qq &&\
    apt-get install -qq man wget sudo vim tmux

RUN apt update

RUN apt install -y cudnn9

RUN yes | pip install --upgrade pip

COPY requirements.txt /home/
WORKDIR /home
RUN yes | pip install -r requirements.txt

# RUN yes | pip install numpy==1.26.4 matplotlib

COPY discriminator.py /home/
COPY generator.py /home/
COPY helper.py /home/
COPY main.py /home/
COPY discriminator.py /home/
COPY config.yaml /home/