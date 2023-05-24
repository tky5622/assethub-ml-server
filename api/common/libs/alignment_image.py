import numpy as np 
# mean+std of hysts-keypoints statistics of rendered front-images
# image size is 512x512

from common.libs.panic3d._util.twodee_v1 import * ; import common.libs.panic3d._util.twodee_v1 as u2d

import io
import tempfile
import requests

## from anime face detector 
import cv2

from anime_face_detector import create_detector

import ipdb
IMAGE = 'https://wuyspkxtjxjlklqkchxr.supabase.co/storage/v1/object/sign/inference_data/7fe65695-a8e0-4b29-8ba8-9256d64904d3/front.png?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJpbmZlcmVuY2VfZGF0YS83ZmU2NTY5NS1hOGUwLTRiMjktOGJhOC05MjU2ZDY0OTA0ZDMvZnJvbnQucG5nIiwiaWF0IjoxNjgzNzA3OTMwLCJleHAiOjFlKzEyMn0.u8F0wh4RK_zmE5rwKHsfWE6IT3YKR6j_Q2eqyG_6uOo&t=2023-05-10T08%3A38%3A50.622Z'
IMAGES = 'https://wuyspkxtjxjlklqkchxr.supabase.co/storage/v1/object/sign/inference_data/7fe65695-a8e0-4b29-8ba8-9256d64904d3/test.png?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJpbmZlcmVuY2VfZGF0YS83ZmU2NTY5NS1hOGUwLTRiMjktOGJhOC05MjU2ZDY0OTA0ZDMvdGVzdC5wbmciLCJpYXQiOjE2ODQ3NDgyMzMsImV4cCI6MTcxNjI4NDIzM30.vzK9GZSFc6U4Uz3AQoEoJ0X1rgGG_Cm456ilmSTkNNk&t=2023-05-22T09%3A37%3A13.155Z'
def imread_web(url):
    # 画像をリクエストする
    res = requests.get(url)
    img = None
    print(res.content, 'fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff')
    # Tempfileを作成して即読み込む

    with tempfile.NamedTemporaryFile(dir='./') as fp:
        fp.write(res.content)
        fp.file.seek(0)
        img = cv2.imread(fp.name)
        I_image = u2d.I(fp.name)
    return img, I_image

## image == I object
def _apply_M(img, M, size=512):
    # ipdb.set_trace()
    return u2d.I(kornia.geometry.transform.warp_affine(
        img.convert('RGBA').bg('w').convert('RGB').t()[None],
        torch.tensor(M).float()[[1,0,2]].T[[1,0,2]].T[None,:2],
        (size,size),
        mode='bilinear',
        padding_mode='fill',
        align_corners=True,
        fill_value=torch.ones(3),
    )).alpha_set(u2d.I(kornia.geometry.transform.warp_affine(
        img.convert('RGBA')['a'].t()[None],
        # img['a'].t()[None],
        # cast error 
        torch.tensor(M).float()[[1,0,2]].T[[1,0,2]].T[None,:2],
        (size,size),
        mode='bilinear',
        padding_mode='fill',
        align_corners=True,
        fill_value=torch.zeros(3),
    ))['r'])




def create_keypoints_anime_face(image_url):
    detector = create_detector('yolov3')
    results = imread_web(image_url)
    image_cv, I_image = results
    preds = detector(image_cv)
    kpts = preds[0]['keypoints']
    M = face_alignment_transform(kpts)
    transformed_image = _apply_M(I_image, M)
    return preds, transformed_image, image_cv, I_image, M


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