FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

RUN apt-get update && \
    apt-get install -y \
    libspatialindex-dev \
    gdal-bin \
    libgdal-dev \
    libgl1-mesa-glx 

RUN mkdir /code

WORKDIR /code
COPY ./ .

RUN pip install -r requirements.txt

CMD [ "python", "/code/yolo_inference.py" ]
