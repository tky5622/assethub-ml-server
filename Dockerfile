# copied from panic3d dockerfile

FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
ENV NVIDIA_DRIVER_CAPABILITIES all

RUN rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub


RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        software-properties-common curl vim git zip unzip unrar p7zip-full wget cmake \
        apache2 openssl libssl-dev

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        libwebp-dev libcairo2-dev libjpeg-dev libgif-dev \
        libboost-all-dev libopencv-dev libwxgtk3.0-gtk3-dev \
        ffmpeg libgl1-mesa-glx libsm6 libxext6 libxrender-dev libx11-xcb1 \
        mesa-utils xauth xorg openbox xvfb



WORKDIR /usr/src/

COPY ./api /usr/src/api
# COPY ./environment.yml /usr/src/app
# COPY ./requirement.txt


RUN conda install \
    'click>=8.0' \
    'scipy' \
    'ninja=1.10.2' \
    'matplotlib=3.4.2' \
    'imageio=2.9.0' \
    'scikit-learn==1.0.2' \
    'scikit-image==0.19.2' \
    'imagesize==1.3.0' \
    'jupyterlab==3.3.2' \
    'flask'

RUN conda install --force-reinstall \
    'pillow==9.0.1'

RUN conda install -c conda-forge \
    'patool==1.12' \
    'addict==2.4.0' \
    'igl==2.2.1' \
    'meshplot==0.4.0' \
    'wandb==0.12.19'\
    'supabase'

RUN pip install \
    'imgui==1.3.0' \
    'glfw==2.2.0' \
    'pyopengl==3.1.5' \
    'imageio-ffmpeg==0.4.3' \
    'pyspng==0.1.0' \
    'mrcfile==1.3.0' \
    'tensorboard==2.9.1' \
    'pyunpack==0.3' \
    'pygltflib==1.15.2' \
    'kornia==0.6.5' \
    'cupy-cuda113==10.5.0' \
    'einops==0.4.1'

RUN pip install \
    'pytorch-lightning==1.6.5' \
    'torchmetrics==0.9.3'

RUN pip install \
    'umap-learn==0.5.3' \
    'lpips==0.1.4'

RUN pip install \
    'git+https://github.com/openai/CLIP.git@d50d76d'

RUN pip install \
    'chamferdist==1.0.0'

RUN pip install \
    'opencv-contrib-python==4.5.4.60'
RUN pip install \
    'opencv-python==4.5.4.60'

RUN pip install \
    'markupsafe==2.0.1' \
    'ipdb' \
    'trimesh'

# for anime face detector
RUN pip install openmim
RUN mim install mmcv-full==1.7.0
RUN mim install mmdet==2.28.2
RUN mim install mmpose==0.29.0

RUN pip install anime-face-detector

RUN pip uninstall numpy
RUN pip install numpy

# copied from panic3d dockerfile by here

#  setting for flask


ENV PYTHONPATH '/usr/src/api:/usr/src/api/common/libs/panic3d:/usr/src/api/common/libs/panic3d/_train/eg3dc/src/'
ENV FLASK_APP "run.py"
# ENV IMAGE_URL "/storage/images"
ENV FLASK_ENV "development"
ENV PROJECT_DN="/usr/src/api/common/libs/panic3d"

## supabase
ENV SUPABASE_URL="https://wuyspkxtjxjlklqkchxr.supabase.co"
ENV SUPABASE_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Ind1eXNwa3h0anhqbGtscWtjaHhyIiwicm9sZSI6ImFub24iLCJpYXQiOjE2ODM3MDIwMTIsImV4cCI6MTk5OTI3ODAxMn0.t-KoM175UyI6yUMUgN4g3F6-Lz3W-KWUAFO2hGkkgN8"


EXPOSE 5000 

# CMD ["flask", "run", "-h", "0.0.0.0"]

