import glob
import os.path

import torch
import json
import numpy as np
from tqdm import tqdm
from deepfool import deepfool
from matplotlib import pyplot as plt

from nn import NoisyNN
from data import getDS, getInvTransforms
from utils import update_cfg_from_models

def does_nets_agree(image, nets, num_classes = 10):
    N = len(nets)

    label = None
    for n in range(N):
        with torch.no_grad():
            f_image = nets[n].forward(image[None, :]).data.cpu().numpy().flatten()
        I = (np.array(f_image)).flatten().argsort()[::-1]

        I = I[0:num_classes]
        if label is None:
            label = I[0]
        elif label != I[0]:
            return False
    return True


def create_adv_sample(cfg):
    data = getDS(cfg, train = True, adversial = False)
    weights_folder = os.path.join(cfg['models_path'], '')

    weights_paths = glob.glob(weights_folder + "UNTAGGED*")
    nets = []
    for i, path in enumerate(weights_paths):
        model = NoisyNN(cfg)
        model.loadWeights(path)
        model.nn.eval()
        nets.append(model.nn)

    for idx_in_ds, sample in tqdm(enumerate(data)):
        im = sample[0]
        agree = does_nets_agree(im, nets)
        if agree:
            break
    if not agree:
        raise Exception("Couldn't find an image")
    else:
        print("Image index = " + str(idx_in_ds))

    r_total, _, label_orig, label_pert, pert_image, found = deepfool(im, nets)
    if not found:
        raise Exception("Failed to find pertubated image")

    return label_orig, label_pert, pert_image, im, r_total, idx_in_ds


def build_sample_prefix(cfg):
    name = cfg['nn_type'] + "_" + "norm-" + str(cfg['normalize']) + "_" +\
           cfg['ds_name'] + "_" + "epochs-" + str(cfg['epochs']) + "_" +\
           "clip-" + str(cfg['clip_val']) + "_" + "m-" + str(cfg['num_models']) + "_"
    return name


def main(cfg):
    models = glob.glob(os.path.normpath(cfg['models_path']) + "/UNTA*")
    assert len(models) == cfg['num_models']

    update_cfg_from_models(cfg, cfg['models_path'])

    ret = create_adv_sample(cfg)
    label_orig, label_pert, pert_image, im, r_total, idx_in_ds = ret

    prefix = os.path.normpath(cfg['adv_sample_folder']) + "/" + build_sample_prefix(cfg)
    torch.save(pert_image.reshape(pert_image.shape[1:]), prefix + "image.pkl")
    print("saved image to " + prefix + "image.pkl")

    cfg.update({
               'adv_label': int(label_pert[0]),
               'orig_label': int(label_orig),
               'img_idx': int(idx_in_ds),
               'most_common_labels' : [int(x) for x in label_pert],
             })
    with open(prefix + "label.json", 'w') as wf:
        json.dump(cfg, wf, indent=1)

    trans = getInvTransforms(cfg)
    pert_image = trans(pert_image.cpu().reshape(pert_image.shape[1:]))
    im = trans(im.cpu())
    r_total_im = trans(torch.tensor(r_total.reshape(r_total.shape[1:])))

    plt.figure()
    plt.imshow(pert_image)
    plt.title("Orignal: " + str(label_orig) + " New: " + str(label_pert))
    plt.show(block=False)
    plt.figure()
    plt.imshow(r_total_im)
    plt.title("Changes in Image")
    plt.show(block=False)
    plt.figure()
    plt.imshow(im)
    plt.title("Orignal Image")
    plt.show()
    return r_total

def getParser():
    import argparse
    parser = argparse.ArgumentParser(description = "Build adversial sample")
    parser.add_argument("--models_path", type = str, required = True)
    parser.add_argument("--adv_sample_folder", type = str, help = "where to"\
                        "save the adversial sample", required = True)
    parser.add_argument("--num_models", type = int, help = "number of models"\
                         "to use for adversial sample", required = True)
    return parser

if __name__ == "__main__":
    parser = getParser()
    args = parser.parse_args()

    cfg = args.__dict__
    main(cfg)
