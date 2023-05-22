
docker build .
docker run --rm --gpus=all --ipc=host --net=host -it 55b22f751f57 (image is)

docker run --rm --gpus=all -p 5000:5000 -it 4d82dc6bf5a1 flask run --host=0.0.0.0

download models zip file and extract models dicretory
https://drive.google.com/file/d/1elk7nTQWhgzig9w05EPgzNs1-NwrP5Jv/view?usp=share_link
flask run --host=0.0.0.0
