import os
import unittest
import glob
import json
import pickle
import numpy as np
import sys
import torch

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.append(os.path.normpath(src_path))
from adv_sample import main
from data import getAdvSample, getDS
from nn import NoisyNN

class TestAdvSample(unittest.TestCase):
    #@unittest.skip("skipping")
    def test_full(self):
        cfg = {
            'models_path' : os.path.normpath(os.path.abspath('misc/trained_model')),
            'adv_sample_folder' : os.path.normpath(os.path.abspath('misc/adv_samples')),
            'num_models' : 2,
        }
        r_total = main(cfg)
        path = glob.glob(cfg['adv_sample_folder'] + "/*.json")[0]
        with open(path, 'r') as rf:
           jf = json.load(rf)
        img_idx = jf['img_idx']
        adv_label = jf['adv_label']
        orig_label = jf['adv_label']
        cfg['adv_sample_img'] = path[:-10] + "image.pkl"
        cfg['ds_name'] : 'MNIST'
        cfg['normalize'] : True

        loaded_orig_img, loaded_orig_label = getAdvSample(cfg, False)
        self.assertEqual(orig_label, loaded_orig_label)

        orig_ds = getDS(cfg, True, False)
        loaded_orig_img_2, loaded_orig_label_2 = orig_ds[img_idx]
        self.assertEqual(loaded_orig_label_2, loaded_orig_label)
        self.assertEqual(torch.sum(torch.abs(loaded_orig_img_2 - loaded_orig_img)), 0)

        im = torch.load(cfg['adv_sample_img'])
        self.assertAlmostEqual(torch.sum(torch.abs(im - loaded_orig_img - r_total)).numpy()
                               , 0, places = 4)

        cfg['nn_type'] = 'LeNet5'
        path = glob.glob(cfg['models_path'] + "/UNT*")[0]
        model = NoisyNN(cfg)
        model.loadWeights(path)
        pred = model.nn(im[None, :])
        label = np.argmax(pred.detach().numpy())
        self.assertEqual(label, adv_label)