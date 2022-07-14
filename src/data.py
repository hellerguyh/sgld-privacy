import torch
import torchvision
from torch.utils.data import DataLoader
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
import json

import warnings

from utils import advImgToLabelPath

CODE_TEST = False

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

class TagMNIST(torchvision.datasets.MNIST):
    def __init__(self, *args, **kwargs):
        adv_sample_img_path = kwargs['adv_sample_img']
        kwargs.pop('adv_sample_img')
        adv_sample_label_path = advImgToLabelPath(adv_sample_img_path)

        super(TagMNIST, self).__init__(*args, **kwargs)

        self.adv_img = torch.load(adv_sample_img_path)
        with open(adv_sample_label_path, 'r') as rf:
            jf = json.load(rf)
            self.orig_label = int(jf['orig_label'])
            self.adv_idx = int(jf['img_idx'])

    def __getitem__(self, index):
        if index == self.adv_idx:
            img, target = self.adv_img, self.orig_label
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target
        else:
            return super(TagMNIST, self).__getitem__(index)


def getTransforms(cfg):
    normalize = cfg['normalize']
    ds_name = cfg['ds_name']
    assert ds_name == "MNIST"
    trans = [torchvision.transforms.Resize((32, 32)),
             torchvision.transforms.ToTensor()]
    if normalize:
        nrm = torchvision.transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD)
        trans.append(nrm)
    return torchvision.transforms.Compose(trans)


def getInvTransforms(cfg):
    normalize = cfg['normalize']
    ds_name = cfg['ds_name']
    assert ds_name == "MNIST"
    trans = []
    if normalize:
        f_s = lambda x: 1 / x
        f_m = lambda x: -x
        nrm = [torchvision.transforms.Normalize(mean=[0], std=list(map(f_s, MNIST_STD))),
               torchvision.transforms.Normalize(mean=list(map(f_m, MNIST_MEAN)), std=[1])]
        trans.extend(nrm)
    trans.append(torchvision.transforms.ToPILImage())
    return torchvision.transforms.Compose(trans)

def getDS(cfg, train, adversial):
    ds_name = cfg['ds_name']
    assert ds_name == "MNIST"

    if adversial:
        adv_sample_img = cfg['adv_sample_img']
        ds = TagMNIST(root = '../dataset/',
                      train = train,
                      download = True,
                      transform = getTransforms(cfg),
                      adv_sample_img = adv_sample_img)
    else:
        ds = torchvision.datasets.MNIST(root = '../dataset/',
                                        train = train,
                                        download = True,
                                        transform = getTransforms(cfg))
    return ds

def getDL(cfg, train, batch_size, adversial, use_batch_sampling):
    ds = getDS(cfg, train, adversial)

    if CODE_TEST:
        for i in range(10):
            warnings.warn("Running in CODE_TEST mode")
        subset = list(range(0,len(ds), int(len(ds)/1000)))
        ds = torch.utils.data.Subset(ds, subset)

    if use_batch_sampling:
        assert train, "Batch sampling should only happen in training"
        sample_rate = float(batch_size/len(ds))
        bsampler =  UniformWithReplacementSampler(
                                                 num_samples=len(ds),
                                                 sample_rate=sample_rate,
                                                 generator=None,
                                                 )
        loader = torch.utils.data.DataLoader(ds, num_workers = 1,
                                             pin_memory = True,
                                             batch_sampler = bsampler)
        loader.sample_rate = sample_rate
    else:
        NW = 4
        torch.set_num_threads(NW)
        loader = torch.utils.data.DataLoader(ds, batch_size = batch_size,
                                             shuffle = train, num_workers = NW,
                                             pin_memory = True)
    loader.ds_size = len(ds)
    return loader

def getAdvSample(cfg, adversial):
    # Using the dataset class since I want data to be loaded exactly how it
    # does in training
    adv_sample_img_path = cfg['adv_sample_img']
    adv_sample_label_path = advImgToLabelPath(adv_sample_img_path)
    with open(adv_sample_label_path, 'r') as rf:
        jf = json.load(rf)
        index = int(jf['img_idx'])

    ds = getDS(cfg, True, adversial)
    img, label = ds[index]
    return img, label

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    cfg = {
            'ds_name' : 'MNIST',
            'adv_sample_img' : '../adv_samples/lenet5_mnist_bs4_clipped0_m30_image.pkl',
            'normalize': True,
          }
    non_adv_ds = getDS(cfg, True, False)
    image, label = non_adv_ds[0]
    plt.figure()
    plt.imshow(image[0].cpu().numpy())
    plt.title("image at place 0: " + str(label))
    plt.show()
    adv_image, adv_label = getAdvSample(cfg, True)
    plt.figure()
    plt.imshow(adv_image[0].cpu().numpy())
    plt.title("adversial image " + str(adv_label))
    plt.show()
    replaced_image, replaced_label = getAdvSample(cfg, False)
    plt.figure()
    plt.imshow(replaced_image[0].cpu().numpy())
    plt.title("replaced image " + str(replaced_label))
    plt.show()
    dl = getDL(cfg, train = True, batch_size = 4, adversial = True,
               use_batch_sampling = False)
    for i, (img, label) in enumerate(dl):
        plt.figure()
        plt.imshow(img[0][0].cpu().numpy())
        plt.title("DL Image " + str(label))
        plt.show()
        break
    dl = getDL(cfg, train = True, batch_size = 4, adversial = True,
               use_batch_sampling = True)
    for i, (img, label) in enumerate(dl):
        plt.figure()
        plt.imshow(img[0][0].cpu().numpy())
        plt.title("DL Image " + str(label))
        plt.show()
        break
