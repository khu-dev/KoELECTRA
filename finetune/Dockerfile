FROM nvcr.io/nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04
LABEL maintainer="nlkey2022@gmail.com"

RUN DEBIAN_FRONTEND=noninteractive apt-get -qq update \
 && DEBIAN_FRONTEND=noninteractive apt-get -qqy install curl python3-pip git \
 && rm -rf /var/lib/apt/lists/*

ARG PYTORCH_WHEEL="https://download.pytorch.org/whl/cu101/torch-1.6.0%2Bcu101-cp36-cp36m-linux_x86_64.whl"
ARG MODEL="1YrkwfM-0VBCJaa9NYaXUQPODdGPsmQY4"

RUN git clone https://github.com/khu-dev/KoELECTRA.git /app/KoELECTRA \
    && cd /app/KoELECTRA/finetune

WORKDIR /app/KoELECTRA/finetune

RUN pip3 install ${PYTORCH_WHEEL} gdown
RUN gdown https://drive.google.com/uc?id=${MODEL} -O model/

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3", "run_serve.py"]

