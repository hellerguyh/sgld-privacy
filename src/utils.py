import os.path

import torch
from time import gmtime, strftime
import pickle
import json
import random
import string
import glob

def acc_score_fn(outputs, labels):
    _, preds = torch.max(outputs, 1)
    corr = torch.sum(preds == labels.data)
    corr_sum = corr.detach().item()
    return corr_sum

def l2error_score_fn(outputs, labels):
    err = torch.sum((outputs - labels)**2).detach().item()
    return err

def l1error_score_fn(outputs, labels):
    err = torch.sum(torch.abs(outputs - labels)).detach().item()
    return err

def _saveMeta(path, model_id, meta):
    with open(path + 'meta_' + model_id + ".pkl", 'wb') as wf:
        pickle.dump(meta, wf)

def _loadMeta(path):
    with open(path, 'rb') as rf:
        meta = pickle.load(rf)
    return meta

def _saveParams(path, model_id, cfg):
    with open (path + "params_" + model_id + ".json", 'w') as wf:
               json.dump(cfg, wf, indent = 1)
    

'''
getID() - create a special ID for each run
@tag: does it based on TAGed dataset

Return: string which is a special ID
'''
def getID(tag):
    RND_LEN = 8
    if tag:
        id_s = 'TAGGED_'
    else:
        id_s = 'UNTAGGED_'
    id_s += strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    id_s += '_'
    id_s += ''.join(random.choices(string.ascii_uppercase + string.digits,
                                   k=RND_LEN))
    return id_s

def update_cfg_from_models(cfg, path):
    m_path = os.path.normpath(path) + "/params_*"
    model_params = glob.glob(m_path)
    with open(model_params[0], 'r') as rf:
        p = json.load(rf)
    # update w.o overwrite
    tmp = cfg.copy()
    cfg.update(p)
    cfg.update(tmp)

def advImgToLabelPath(img_path):
    assert img_path[-10:] == "_image.pkl", "adversial sample path should end"\
                                          "with _image.pkl but ends with " + img_path[-10 : ]
    path_prefix = img_path[:-10]
    return path_prefix + "_label.json"

