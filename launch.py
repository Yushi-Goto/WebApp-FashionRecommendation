# DESCRIPTION OF launch.py
#
# Webアプリを起動させます．
# また，クライアントのリクエストに応じて，ルーティングを行います．

from flask import Flask, render_template, request
import os
import torch
import argparse
import models

app = Flask(__name__)

device = torch.device('cpu')
param = {'device': device, 'mode': 'test', 'label_features_path': './label_features.npy',
                        'model_path': './FRModel.pht', 'param_path': './parameters.pickle'}
model = models.FRModel(param)

@app.route("/")
def upload_file():
    return render_template("select.html")

@app.route("/result", methods=["POST"])
def show_images():
    if request.method == 'POST':
        file = request.files['file']
        file_path = './images/test_img.png'
        file.save(file_path)
        recom_img_paths = model.test('choose', file_path, 3)
    return render_template("result.html", img_path_list=recom_img_paths)

if __name__ == '__main__':
    app.run(debug=True)
