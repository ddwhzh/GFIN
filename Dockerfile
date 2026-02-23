FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel

LABEL maintainer="ddwhzh@163.com"
LABEL description="GFIN: Human-Object Interaction Detection via Global Context and Pairwise-level Fusion"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        git \
        vim \
        wget \
        libgtk2.0-dev \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/gfin

COPY requirements.txt /workspace/gfin/
RUN pip install --no-cache-dir -r requirements.txt

COPY detr/ /workspace/gfin/detr/
COPY pocket/ /workspace/gfin/pocket/
COPY gfin_module/ /workspace/gfin/gfin_module/
COPY add_on/ /workspace/gfin/add_on/
COPY hicodet/*.py /workspace/gfin/hicodet/
COPY hicodet/*.json /workspace/gfin/hicodet/
COPY vcoco/*.py /workspace/gfin/vcoco/
COPY vcoco/*.json /workspace/gfin/vcoco/
COPY tools/ /workspace/gfin/tools/
COPY experiments/ /workspace/gfin/experiments/

COPY main.py gfin.py interaction_head.py ops.py utils.py \
     upt.py upt_interaction_head.py inference.py vcoco_test.py \
     .env.example \
     /workspace/gfin/

RUN mkdir -p /workspace/gfin/logs \
             /workspace/gfin/checkpoints \
             /workspace/gfin/outputs_img

EXPOSE 6006

ENTRYPOINT ["python"]
CMD ["main.py", "--help"]
