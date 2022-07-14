import torch
import glob
import json
import numpy as np
import copy
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

from utils_dp import get_emp_eps, get_eps_lower_bound, plotEpsLB, getStats
from utils import _loadMeta, update_cfg_from_models, advImgToLabelPath

'''
MetaDS - Dataset used to handle meta-data

Should be suitable for working with sklearn - i.e. provide the whole data
at once)
'''
class MetaDS(object):
    def __init__(self, cfg, num_models, path, meta):
        self.cfg = cfg
        self.num_models = num_models
        if meta is None:
            self.meta = self.collectMeta(path)
        else:
            self.meta = meta

    def train_test_split(self):
        test_size = self.cfg['dry_run_test_ratio']
        meta_train, meta_test = train_test_split(self.meta,
                                                 test_size = test_size,
                                                 random_state = 0)
        return MetaDS(self.cfg, -1, None, copy.deepcopy(meta_train)),\
               MetaDS(self.cfg, -1, None, copy.deepcopy(meta_test))

    def collectMeta(self, path):
        num_models = self.num_models
        path = os.path.normpath(path) + "/"
        tagged_l = glob.glob(path + "meta_TAGGED*")
        untagged_l = glob.glob(path + "meta_UNTAGGED*")

        selection = np.random.randint(0, 2, num_models)
        ti = 0
        ui = 0
        metadata = []
        for s in selection:
            if s == 0:
                d = _loadMeta(untagged_l[ui])
                ui += 1
            else:
                d = _loadMeta(tagged_l[ti])
                ti += 1
            metadata.append(d)
        print("Using ", ui, " untagged models")
        print("Using ", ti, " tagged models")

        return metadata

    def getMalPred(self, epoch):
        ds_size = len(self)
        feature_indexes = self.cfg['feature_indexes']
        X = np.zeros((ds_size, len(feature_indexes)))
        Y = np.zeros(ds_size)
        for i, sample in enumerate(self.meta):
            x_prop1 = torch.tensor(sample['mal_pred_arr'][epoch]).numpy()
            x_prop2 = torch.tensor(sample['nonmal_pred_arr'][epoch]).numpy()
            for k, j in enumerate(feature_indexes):
                if j < 9:
                    X[i][k] = x_prop1[j]
                else:
                    X[i][k] = x_prop2[j - 10]
            if 'UNTAGGED' in sample['model_id']:
                Y[i] = False
            else:
                Y[i] = True
        return X, Y

    def __len__(self):
        return len(self.meta)

    def getField(self, f_name, epoch, train_or_vld):
        X = np.zeros(len(self))
        for i, sample in enumerate(self.meta):
            X[i] = sample[f_name][train_or_vld][epoch]
        return X

def calcEpsGraph(cfg):
    epochs = cfg['epochs']
    delta = cfg['delta']
    train_path = os.path.normpath(cfg['models_path'] + '/train/')
    train_ds = MetaDS(cfg, cfg['num_train_models'], train_path, None)
    if cfg['dry_run']:
        train_ds, test_ds = train_ds.train_test_split()
    else:
        test_path = os.path.normpath(cfg['models_path'] + '/test/')
        test_ds = MetaDS(cfg, cfg['num_test_models'], test_path, None)

    emp_eps_arr = [0]
    eps_lb_arr = [0]
    score_arr = [np.mean(test_ds.getField('score_arr', 0, 'val'))]

    for epoch in range(1, epochs + 1):
        X_train, Y_train = train_ds.getMalPred(epoch)
        X_test, Y_test = test_ds.getMalPred(epoch)

        clf = make_pipeline(StandardScaler(),
                            SGDClassifier(max_iter=1000, tol=1e-3))
        clf.fit(X_train, Y_train)

        P = clf.predict(X_test)
        FN_rate, FP_rate, FN, FP, pos, negs = getStats(P, Y_test)

        emp_eps = get_emp_eps(FN_rate, FP_rate, delta)
        emp_eps_arr.append(emp_eps)

        eps_lb = get_eps_lower_bound(FN, FP, pos, negs, delta)
        eps_lb_arr.append(eps_lb)

        acc = test_ds.getField('score_arr', epoch, 'val')
        score_arr.append(np.mean(acc))

    plt.plot(emp_eps_arr, label = 'emp_eps')
    plt.plot(eps_lb_arr, label = 'lower bound')
    plt.legend()
    plt.savefig(os.path.normpath(cfg['result_path']) + "/" + 'empirical_epsilon.png')
    plotEpsLB(eps_lb_arr, score_arr, cfg)


def getParser():
    import argparse
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--dry_run_test_ratio", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--models_path", type=str, required=True)
    parser.add_argument("--num_train_models", type=int, required=True)
    parser.add_argument("--num_test_models", type=int, required=True)
    parser.add_argument("--feature_indexes", type=int, nargs="+")
    parser.add_argument("--adv_sample_img", type=str)
    parser.add_argument("--result_path", type=str, required=True)
    return parser

def main():
    parser = getParser()
    args = parser.parse_args()
    cfg = {'models_path' : args.models_path}
    update_cfg_from_models(cfg, os.path.normpath(cfg['models_path']) + '/train/' )

    if args.feature_indexes is None:
        label_path = advImgToLabelPath(args.adv_sample_img)
        with open(label_path, 'r') as rf:
            jf = json.load(rf)
            adv_label = int(jf['adv_label'])
            orig_label = int(jf['orig_label'])
        cfg['feature_indexes'] = [adv_label, orig_label, adv_label + 10,
                                  orig_label + 10]
    for arg in args.__dict__.keys():
        if args.__dict__[arg] is not None:
            cfg.update({arg : args.__dict__[arg]})

    calcEpsGraph(cfg)
    path = os.path.normpath(cfg['result_path']) + "/prediction_params.json"
    with open(path, 'w') as wf:
        json.dump(cfg, wf, indent=1)

if __name__ == "__main__":
    main()
