import os
import torch
import numpy
import cv2
import argparse
import models

def parse_args():
    """Command-line argument parser"""
    parser = argparse.ArgumentParser(description='Fashion Recommendation System')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], help='select mode (train or test)')
    parser.add_argument('--img-size', type=int, help='Image size for input to network')
    parser.add_argument('--dataset-path', default='./dataset', help='dataset root path (default: ./dataset)')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size (default: 64)')
    parser.add_argument('--num-fout1', type=int, default=128, help='output channel of first conv layer (default: 128)')
    parser.add_argument('--num-fout2', type=int, default=64, help='output channel of second conv layer (default: 64)')
    parser.add_argument('--num-fout3', type=int, default=32, help='output channel of third conv layer (default: 32)')
    parser.add_argument('--num-features', type=int, default=512, help='Number of dimensions of feature vector(or space) (default: 512)')
    parser.add_argument('--num-classes', type=int, default=4, help='Number of classes (default: 4)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--label-features-path', default='./label_features.npy', help='save label features path (default: ./label_features.npy)')
    parser.add_argument('--model-path', default='./FRModel.pht', help='save model path (default: ./FRModel.pht)')
    parser.add_argument('--param-path', default='./parameters.pickle', help='save parameter path (default: ./parameters.pickle)')
    parser.add_argument('--test-img-type', type=str, choices=['take', 'choose'], help='select test image type (take or choose)')
    parser.add_argument('--test-img-path', default='./images/test_img.png', help='test image path (default: ./images/test_img.png)')
    parser.add_argument('--genre-option', type=int, default=0, help='Number representing the fashion genre. (test mode)')
    parser.add_argument('--num-recom', type=int, default=3, help='number of recommended fashions (default: 3)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

    return parser.parse_args()


def main():
    args = parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    if args.mode == 'train':
        param = {'device': device, 'mode': args.mode, 'label_features_path': args.label_features_path, 'model_path': args.model_path,
            'param_path': args.param_path, 'img_size': args.img_size, 'dataset_path': args.dataset_path, 'batch_size': args.batch_size,
            'num1': args.num_fout1, 'num2': args.num_fout2, 'num3': args.num_fout3, 'num_features': args.num_features,
            'num_classes': args.num_classes, 'lr': args.learning_rate}
        model = models.FRModel(param)
        model.train(args.epochs)

    elif args.mode == 'test':
        param = {'device':device, 'mode':args.mode, 'label_features_path': args.label_features_path,
                                                            'model_path':args.model_path, 'param_path':args.param_path}
        model = models.FRModel(param)
        _ = model.test(args.test_img_type, args.test_img_path, args.num_recom)

if __name__ == '__main__':
    main()
