FROM continuumio/miniconda3
LABEL maintainer="nclgbd"
# Install base utilities
RUN apt-get update && \
    apt-get install -y build-essential wget  && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY . /thesis-work
RUN conda init bash
RUN pip install -r /thesis-work/requirements.txt
RUN pip install -e /thesis-work