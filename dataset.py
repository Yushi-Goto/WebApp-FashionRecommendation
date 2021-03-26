import os
import numpy
import torch
import cv2


class FRDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, transform=None):
        self.transform = transform

        self.dataset_dirs = []
        for d in os.listdir(dataset_path):  # dataset_path内のディレクトリを取得．
            if os.path.isdir(dataset_path + '/' + d):  # ディレクトリじゃないものを除外(.DSStoreとか)
                self.dataset_dirs.append(d)

        # facesおよびlabelsの読み込み処理
        self.faces = []
        self.labels = []
        for d in self.dataset_dirs:
            file_list = []
            for file in os.listdir(dataset_path + '/' + d):  # 各ディレクトリ内の画像ファイルの取得
                if not file.startswith('.'):  # 隠しファイルの除去
                    file_list.append(file)

            for f in file_list:  # self.facesとself.labelsへのファイルの振り分け
                fname_without_ext = os.path.splitext(f)[0]
                # self.facesに顔画像を追加
                if fname_without_ext == 'face':
                    file_path = dataset_path + '/' + d + '/' + f
                    self.faces.append(cv2.imread(file_path))
                # self.labelsにラベルを追加
                elif fname_without_ext == 'fashion_0':
                    self.labels.append(0)
                elif fname_without_ext == 'fashion_1':
                    self.labels.append(1)
                elif fname_without_ext == 'fashion_2':
                    self.labels.append(2)
                elif fname_without_ext == 'fashion_3':
                    self.labels.append(3)

    def __len__(self):
        return len(self.dataset_dirs)

    def __getitem__(self, idx):
        face = self.faces[idx]
        lbl = self.labels[idx]

        if self.transform:  # 画像の前処理
            face = self.transform(face)
        lbl = torch.tensor(lbl)

        return {'face':face, 'lbl': lbl}
