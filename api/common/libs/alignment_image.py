import numpy as np 
# mean+std of hysts-keypoints statistics of rendered front-images
# image size is 512x512


import io
import tempfile
import requests

## from anime face detector 
import cv2

from anime_face_detector import create_detector

import ipdb

IMAGE = 'https://wuyspkxtjxjlklqkchxr.supabase.co/storage/v1/object/sign/inference_data/7fe65695-a8e0-4b29-8ba8-9256d64904d3/front.png?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJpbmZlcmVuY2VfZGF0YS83ZmU2NTY5NS1hOGUwLTRiMjktOGJhOC05MjU2ZDY0OTA0ZDMvZnJvbnQucG5nIiwiaWF0IjoxNjgzNzA3OTMwLCJleHAiOjFlKzEyMn0.u8F0wh4RK_zmE5rwKHsfWE6IT3YKR6j_Q2eqyG_6uOo&t=2023-05-10T08%3A38%3A50.622Z'

def imread_web(url):
    # 画像をリクエストする
    res = requests.get(IMAGE)
    img = None
    print(res.content, 'fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff')
    # Tempfileを作成して即読み込む

    with tempfile.NamedTemporaryFile(dir='./') as fp:
        fp.write(res.content)
        fp.file.seek(0)
        img = cv2.imread(fp.name)
    return img



def create_keypoints_anime_face(image_url):
    detector = create_detector('yolov3')
    image = imread_web(image_url)
    preds = detector(image)
    return preds


### from panic3d issue (fandom alignment)
# Fixed statistics based on previous dataset
x_mu, x_sig = (294.9637, 8.072594)   # center x-coord (+down)
y_mu, y_sig = (255.8569, 1.8546647)  # center y-coord (+right)
a_mu, a_sig = (255.8569, 1.8546647)  # std of keypoints from center


def face_alignment_transform(keypoints):
    # calc stats
    kpts = keypoints
    c = kpts.mean(axis=0)
    s = np.linalg.norm(kpts-c, axis=1).std()
    rands = np.random.normal(size=3)
    cnew = np.asarray([
        x_mu + x_sig*rands[0],
        y_mu, #+ y_sig*rands[1],
    ])
    snew = a_mu #+ a_sig*rands[2]

    # get matrix
    sf = snew / s
    M = np.asarray([  # recenter to new
        [1,0,cnew[0]],
        [0,1,cnew[1]],
        [0,0,1],
    ]) @ np.asarray([  # scale
        [sf,0,0],
        [0,sf,0],
        [0,0,1],
    ]) @ np.asarray([  # center to origin
        [1,0,-c[0]],
        [0,1,-c[1]],
        [0,0,1],
    ])
    return M