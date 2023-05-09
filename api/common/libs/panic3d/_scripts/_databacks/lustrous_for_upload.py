import os
import torch
from torchvision.io import read_image as I

class DatabackendMinna:
    def __init__(self, uploaded_images_path):
        self.uploaded_images_path = uploaded_images_path
        self.bns = self.get_bns()

    def get_bns(self):
        bns = []

        # アップロードされた画像ファイルの読み込み
        for viewfn in os.listdir(self.uploaded_images_path):
            if viewfn.endswith('.png') and viewfn[0] != '_':
                bns.append(f'uploaded/{viewfn}')

        return sorted(bns)

    def __len__(self):
        return len(self.bns)

    def __getitem__(self, index, return_more=False):
        bn = self.bns[index]
        ret = {'bn': bn}

        # 画像の読み込み
        ret['image'] = I(os.path.join(self.uploaded_images_path, f'{bn}.png'))

        if return_more:
            ret.update({'locals': locals()})
        return ret
