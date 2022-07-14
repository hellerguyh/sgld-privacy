import argparse
import json
import torch
import numpy as np
from matplotlib import pyplot as plt
import glob

from utils import update_cfg_from_models
from nn import NoisyNN

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="scatter plot")
    parser.add_argument("--pkl_path", type=str, default=None)
    parser.add_argument("--json_path", type=str, default=None)
    parser.add_argument("--weights_folder", type=str, default=None)
    args = parser.parse_args()
    if args.weights_folder [-1] != "/":
        args.weights_folder += "/"

    cfg = {}
    update_cfg_from_models(cfg, args.weights_folder)
    for postfix in ["TAG", "UNTAG"]:
        weights_paths = glob.glob(args.weights_folder + postfix + "*")
        print(len(weights_paths))
        adv_image = torch.load(args.pkl_path)
        adv_image = adv_image.reshape([1, *adv_image.shape])
        with open(args.json_path, 'r') as rf:
            jf = json.load(rf)
            label_idx = int(jf['orig_label'])
            adv_label_idx = int(jf['adv_label'])
        pred_arr = np.zeros(len(weights_paths))
        pred_arr_adv = np.zeros(len(weights_paths))
        softmax = torch.nn.Softmax(dim=0)
        for i, path in enumerate(weights_paths):
            model = NoisyNN(cfg)
            model.loadWeights(path)
            model.nn.eval()
            pred = model.nn(adv_image).detach()[0]
            pred = softmax(pred)
            pred_arr[i] = pred.numpy()[label_idx]
            pred_arr_adv[i] = pred.numpy()[adv_label_idx]

        plt.scatter(pred_arr_adv, pred_arr, label = postfix)
    plt.legend()
    plt.xlabel("adversial label")
    plt.ylabel("original label")
    plt.show()

#--create_adv_sample --path trained_weights/lenet5/cifar10/lr500e400bs32normalized/createAdvExample/ --nn LeNet5 --dataset CIFAR10 --normalize
#--train_model --path trained_weights/variable/ --nn LeNet5 --dataset CIFAR10 --normalize --cuda_id -1 --lr_factor 500
