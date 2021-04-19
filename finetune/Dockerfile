FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
LABEL maintainer="nlkey2022@gmail.com"

RUN apt update && apt install -y git

ARG MODEL="1YrkwfM-0VBCJaa9NYaXUQPODdGPsmQY4"

RUN git clone https://github.com/khu-dev/KoELECTRA.git /app/KoELECTRA \
    && cd /app/KoELECTRA/finetune

WORKDIR /app/KoELECTRA/finetune

RUN pip install gdown
RUN gdown https://drive.google.com/uc?id=${MODEL} -O model/

RUN pip install -r requirements.txt

ENTRYPOINT ["python3", "run_serve.py"]

