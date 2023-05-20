import numpy as np 
# mean+std of hysts-keypoints statistics of rendered front-images
# image size is 512x512



## from anime face detector 
import cv2

from anime_face_detector import create_detector




def create_keypoints_anime_face(image):
    detector = create_detector('yolov3')
    cv2_image = cv2.imread(image)
    preds = detector(cv2_image)
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