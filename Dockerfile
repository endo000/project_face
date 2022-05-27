FROM python:3.8.13-buster
# FROM dkimg/opencv:4.5.3-ubuntu

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    bzip2 \
    g++ \
    git \
    graphviz \
    libgl1-mesa-glx \
    libhdf5-dev \
    openmpi-bin \
    wget \
    python3-tk && \
    rm -rf /var/lib/apt/lists/*

# Setting up working directory 
RUN mkdir /src
WORKDIR /src
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
# Minimize image size 
RUN (apt-get autoremove -y; \
    apt-get autoclean -y)

COPY . .
CMD ["python", "main.py"]
EXPOSE 5000