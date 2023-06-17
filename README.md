0. download models zip file and extract "models" dicretory
https://drive.google.com/file/d/1elk7nTQWhgzig9w05EPgzNs1-NwrP5Jv/view?usp=share_link

2. docker build .
3. docker run --rm --gpus=all -p 5000:5000 -it "imageId" (like 4d82dc6bf5a1) flask run --host=0.0.0.0

