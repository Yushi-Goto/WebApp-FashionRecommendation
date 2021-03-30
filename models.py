# DESCRIPTION OF models.py
#
# トレーニング（学習）とテスト（推薦）の機能を持ちます．
# また，トレーニングとテストに必要なその他の機能を持ちます．

import os
import numpy
import cv2
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch import optim
import utils.transforms
import dataset
import nets
import metrics
import pickle

class FRModel(object):
    def __init__(self, param):
        super(FRModel, self).__init__()
        self.device = param['device']
        self.label_features_path = param['label_features_path']
        self.model_path = param['model_path']
        self.param_path = param['param_path']

        # トレーニング（学習）の設定
        if param['mode'] == 'train':
            self.img_size = param['img_size']

            # 教師データの顔画像への前処理設定
            self.train_data_transform = transforms.Compose([
                utils.transforms.Resize(param['img_size']),
                utils.transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

            # データセットの設定および準備
            self.dataset_path = param['dataset_path']
            self.batch_size = param['batch_size']
            self.train_dataset = dataset.FRDataset(dataset_path=param['dataset_path'], transform=self.train_data_transform)
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

            # ネットワーク（モデル）の設定および生成
            self.num_fout1 = param['num1']
            self.num_fout2 = param['num2']
            self.num_fout3 = param['num3']
            self.num_features = param['num_features']
            self.num_classes = param['num_classes']
            self.net = nets.FRNet(param['device'], param['img_size'], param['num_features'],
                                                                            param['num1'], param['num2'], param['num3']).to(param['device'])
            self.metric_net = metrics.ArcMarginProduct(in_features=param['num_features'], out_features=param['num_classes']).to(param['device'])

            # トレーニング（学習）時における，学習係数，最適化手法，損失関数の設定
            self.learning_rate = param['lr']
            self.optimizer = optim.Adam([{'params': self.net.parameters()}, {'params': self.metric_net.parameters()}], param['lr'])
            self.criterion = nn.CrossEntropyLoss()

            # トレーニング（学習）時に用いたパラメータ設定の保存
            self.save_train_parameters()

        # テスト（推薦）の設定
        elif param['mode'] == 'test':
            t_param = self.load_train_parameters()
            self.dataset_path = t_param['dataset_path']

            # 対象人物の顔画像への前処理設定
            self.test_data_transform = transforms.Compose([
                utils.transforms.GetFace(),
                utils.transforms.Resize(t_param['img_size']),
                utils.transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

            self.net = nets.FRNet(param['device'], t_param['img_size'], t_param['num_features'],
                                                                    t_param['num1'], t_param['num2'], t_param['num3']).to(param['device'])

    def train(self, epoch):
        """トレーニング（学習）の実行"""
        self.net.train()
        self.metric_net.train()

        label_features = []
        for times in range(epoch):
            running_loss = 0.0

            for idx, data in enumerate(self.train_loader):
                # 入力データ・ラベル
                batch_faces = data["face"].to(self.device)
                batch_labels = data["lbl"].to(self.device)

                # optimizerの初期化 -> 順伝播 -> Lossの計算 -> 逆伝播 -> パラメータの更新
                self.optimizer.zero_grad()
                features = self.net(batch_faces)    # 特徴ベクトルを生成（特徴抽出）
                outputs = self.metric_net(features, batch_labels)   # 距離学習（ArcFace）フェーズ
                # outputs = F.softmax(outputs, dim=1)
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()

                # 最終エポック時，各ラベルの特徴ベクトルをリスト化
                if times == (epoch - 1):
                    features_temp = features.to(torch.device('cpu'))
                    features_numpy = features_temp.detach().numpy().copy()
                    for i, feature in enumerate(features_numpy):
                        label_features.append(feature)

        # 各教師データ顔画像の特徴ベクトルの保存
        self.save_label_features(label_features)

        # 学習済みネットワーク（モデル）の保存
        self.save_model()

        print('finished training')

    def test(self, test_img_type, test_img_path, num_recom):
        """テスト（推薦）の実行"""
        # test画像の読み込み
        if test_img_type == 'take':
            self.save_frame_camera_key(0, test_img_path)
            test_img = cv2.imread(test_img_path)
        elif test_img_type == 'choose':
            test_img = cv2.imread(test_img_path)

        # test画像の前処理
        test_img = self.test_data_transform(test_img)

        # VGG16にtest_imgを入力する際に生じるデータサイズの差異の解消
        test_img = test_img.numpy().copy()  # tensor -> numpy
        test_img = torch.from_numpy(test_img[None, :, :, :]).float()  # 次元追加, numpy -> tensor

        # label = torch.tensor([label])

        # 学習済みネットワーク（モデル）の読み込み
        self.load_model()
        self.net.eval()
        with torch.no_grad():
            # 特徴ベクトルを生成（特徴抽出）
            feature = self.net(test_img.to(self.device))
            # feature = F.softmax(feature, dim=1)

        # 対象人物の特徴ベクトルと教師データの特徴ベクトルを比較し，推薦ファッション画像の保存
        recom_img_paths = self.save_recom_img(num_recom, feature)

        return recom_img_paths

    def save_label_features(self, label_features):
        """Save label features to self.label_features_path"""
        """各教師データ顔画像の特徴ベクトルの保存"""
        path_without_ext = os.path.splitext(self.label_features_path)[0]
        numpy.save(path_without_ext, label_features)

    def save_model(self):
        """Save model to self.model_path"""
        """学習済みネットワーク（モデル）の保存"""
        torch.save(self.net.state_dict(), self.model_path)
        # torch.save(self.net.state_dict(), self.model_path+'Classification')
        # torch.save(self.metric_net.state_dict(), self.model_path+'ArcFace')

    def load_model(self):
        """Load model from self.model_path"""
        """学習済みネットワーク（モデル）の読み込み"""
        self.net.load_state_dict(torch.load(self.model_path, map_location=self.device))

    def save_frame_camera_key(self, device_num, img_path, delay=1, window_name='frame'):
        """Take and save a test image"""
        """対象となる人物画像の撮影および保存"""
        cap = cv2.VideoCapture(device_num)

        if not cap.isOpened():
            return

        while True:
            ret, frame = cap.read()
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(delay) & 0xFF
            if key == ord('c'):
                cv2.imwrite(img_path, frame)
            elif key == ord('q'):
                break

        cv2.destroyWindow(window_name)

    def save_recom_img(self, num_recom, feature):
        """Save recommended fashion images"""
        """推薦ファッション画像の保存"""
        feature = feature.to(torch.device('cpu'))
        feature_numpy = feature.numpy().copy()

        # 対象人物の特徴ベクトルと教師データの特徴ベクトル間の距離を計算
        dist_btwn_features = []
        label_features = self.load_label_features()
        for l_feature in label_features:
            temp = numpy.linalg.norm(l_feature - feature_numpy)
            dist_btwn_features.append(temp)
        print('\n dist_btwn_features')
        print(dist_btwn_features)

        # 推薦ファッションのインデックスの取得
        recom_list = []
        for i in range(num_recom):
            min_value = min(dist_btwn_features)
            min_idx = dist_btwn_features.index(min_value)
            recom_list.append(min_idx)
            dist_btwn_features[min_idx] = 99999999.9
        print('\n recom_list')
        print(recom_list)

        # 推薦ファッション画像の保存先（static）の初期化
        # （フォルダ内のファイル削除）
        for f in os.listdir('./static'):
            os.remove('./static/' + f)

        # 各推薦ファッション画像およびその保存ディレクトリのパスを取得
        recom_img_paths = []
        for i, l in enumerate(recom_list):
            file_list = []
            for file in os.listdir(self.dataset_path + '/' + str(l)):
                if not file.startswith('.'):  # 隠しファイルの除去
                    file_list.append(file)

            # 推薦ファッション画像をstaticディレクトリに保存
            for f in file_list:
                fname_without_ext = os.path.splitext(f)[0]
                if fname_without_ext == 'fashion_0':
                    file_path = self.dataset_path + '/' + str(l) + '/' + f
                    copyfile_path = './static/recom_img{}-{}.png'.format(i+1, fname_without_ext)
                    shutil.copyfile(file_path, copyfile_path)  # 推薦ファッション画像を./staticにコピー
                    recom_img_paths.append(copyfile_path)
                elif fname_without_ext == 'fashion_1':
                    file_path = self.dataset_path + '/' + str(l) + '/' + f
                    copyfile_path = './static/recom_img{}-{}.png'.format(i+1, fname_without_ext)
                    shutil.copyfile(file_path, copyfile_path)
                    recom_img_paths.append(copyfile_path)
                elif fname_without_ext == 'fashion_2':
                    file_path = self.dataset_path + '/' + str(l) + '/' + f
                    copyfile_path = './static/recom_img{}-{}.png'.format(i+1, fname_without_ext)
                    shutil.copyfile(file_path, copyfile_path)
                    recom_img_paths.append(copyfile_path)
                elif fname_without_ext == 'fashion_3':
                    file_path = self.dataset_path + '/' + str(l) + '/' + f
                    copyfile_path = './static/recom_img{}-{}.png'.format(i+1, fname_without_ext)
                    shutil.copyfile(file_path, copyfile_path)
                    recom_img_paths.append(copyfile_path)

        print('saved recommended fashion images')
        print(recom_img_paths)
        return recom_img_paths

    def save_train_parameters(self):
        """Save train parameters """
        """トレーニング（学習）時におけるパラメータの保存"""
        with open(self.param_path, 'wb') as parameters:
            p = {
                'img_size': self.img_size,
                'dataset_path': self.dataset_path,
                'num_features': self.num_features,
                'num1': self.num_fout1,
                'num2': self.num_fout2,
                'num3': self.num_fout3}
            pickle.dump(p, parameters)

    def load_train_parameters(self):
        """Load train parameters"""
        """トレーニング（学習）時におけるパラメータの読み込み"""
        with open(self.param_path, 'br') as parameters:
            param = pickle.load(parameters)
        return param

    def load_label_features(self):
        """Load label features"""
        """教師データ顔画像の特徴ベクトルの読み込み"""
        label_features = numpy.load(self.label_features_path, allow_pickle=True)
        return label_features
