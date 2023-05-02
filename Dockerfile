FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
ENV NVIDIA_DRIVER_CAPABILITIES all

WORKDIR /app
COPY . /app

# A workaround seems to do the trick for me. Add those lines before apt-get update
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list


# Since wget is missing
RUN apt-get update && apt-get install -y wget

#Install MINICONDA
# RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda.sh && \
# 	/bin/bash Miniconda.sh -b -p /opt/conda && \
# 	rm Miniconda.sh

ENV PATH /opt/conda/bin:$PATH

# Install gcc as it is missing in our base layer
RUN apt-get update && apt-get -y install gcc 

RUN conda env create -f environment.yml --force

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Make sure the environment is activated:
RUN echo "Make sure flask is installed:"
RUN python -c "import flask"

EXPOSE 5000

# The code to run when container is started:
ENTRYPOINT ["conda", "run", "-n", "myenv", "python", "src/app.py"]
