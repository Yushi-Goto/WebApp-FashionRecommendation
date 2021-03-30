# DESCRIPTION OF transforms.py
#
# データの前処理を行います．

import torch
import numpy as np
import cv2


class Resize(object):
    """Resize a face image to minimum size"""
    """顔画像のリサイズ処理"""
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, face):
        return cv2.resize(face, (self.output_size, self.output_size))

class ToTensor(object):
    """Convert a ndarray of face image to Tensors."""
    """顔画像をNumPy配列からTensorに変換"""
    def __call__(self, face):
        face = face.transpose((2, 0, 1))
        return torch.from_numpy(face).float()

class GetFace(object):
    """Cut out a face into a square"""
    """画像から顔の切り抜く"""
    def __call__(self, img):
        # 顔の検出
        cascade = cv2.CascadeClassifier("./utils/haarcascades/haarcascade_frontalface_alt.xml")
        face_list = cascade.detectMultiScale(img,scaleFactor=1.5,minNeighbors=1,minSize=(5,5))

        if len(face_list) == 0:
            return img

        elif len(face_list) > 0:
            for rect in face_list:
                x = rect[0]
                y = rect[1]
                width = rect[2]
                height = rect[3]
                dst = img[y - 10:y + height, x:x + width +10]

            cv2.imwrite('./images/input_face_img.png', dst)
            return dst
