
import torch

import numpy as np
import sys
import os

sys.path.append('..')

import random
from tqdm import trange

import dgl
from methods.GPAWP.utils.data_loader import data_loader,data_loader_lp
from methods.GPAWP.utils.data import load_data_IMDB

def few_shot_split_nodelevel(dl, tasknum, trainshot, valshot, labelnum, seed=0, drop=False):
    train = []
    val = []
    test = []
    train_label = []
    val_label = []
    testindex = dl.labels_test['mask'].nonzero()[0]

    if drop:
        labelnum = labelnum - 1

    # nodenum = dl.nodes["total"]
    # nodenum = 4278      #imdb 4278 acm 3025
    nodenum = dl.nodes['count'][0]
    random.seed(seed)

    for count in range(tasknum):
        index = random.sample(range(0, nodenum), nodenum)
        trainindex = []
        valindex = []
        traincount = torch.zeros(labelnum)
        valcount = torch.zeros(labelnum)
        trainlabel = []
        vallabel = []

        for i in index:
            if not dl.labels_train["mask"][i]:
                continue
            if dl.labels_train["mask"][i]:
                label_vec = dl.labels_train["data"][i]
                if drop:
                    if label_vec[labelnum] == 1:
                        continue

                label = np.argmax(label_vec)
            if traincount[label - 1] < trainshot:
                trainindex.append(i)
                traincount[label - 1] += 1
                trainlabel.append(label)
            elif valcount[label - 1] < valshot:
                valcount[label - 1] += 1
                valindex.append(i)
                vallabel.append(label)

        train.append(trainindex)
        val.append(valindex)
        # test.append(testindex)
        train_label.append(trainlabel)
        val_label.append(vallabel)

    return train, val, testindex, train_label, val_label

train_config={
    "trainshot": 1,
    "valshot": 1,
    "labelnum": 7,
    "tasknum": 100,
    "save_graph_path": "../GPAWP/data",
    "save_fewshot_path":"../GPAWP/data/Fewshot/Freebase_100",
    "seed":0,
    "drop": False,
    "dataset": 'Freebase_100'
}
if __name__ == "__main__":
    for i in range(1, len(sys.argv), 2):
        arg = sys.argv[i]
        value = sys.argv[i + 1]

        if arg.startswith("--"):
            arg = arg[2:]
        if arg not in train_config:
            print("Warning: %s is not surported now." % (arg))
            continue
        train_config[arg] = value
        try:
            value = eval(value)
            if isinstance(value, (int, float)):
                train_config[arg] = value
        except:
            pass

    if train_config["dataset"]=="IMDB":
        _,_,_,_,dl= load_data_IMDB('../GPAWP/data/IMDB')
    else:
        dl = data_loader('../GPAWP/data/Freebase_100')

    trainset,valset,testset,train_label, val_label=few_shot_split_nodelevel(dl,train_config["tasknum"],train_config["trainshot"],
                                                     train_config["valshot"],train_config["labelnum"],
                                                     train_config["seed"],train_config["drop"])
    trainset = np.array(trainset)
    valset = np.array(valset)
    testset = np.array(testset)
    train_label = np.array(train_label)
    val_label = np.array(val_label)
    fewshot_dir=os.path.join(train_config["save_fewshot_path"],"%sshots%stasks" %
                             (train_config["trainshot"],train_config["tasknum"]))

    if os.path.exists(train_config["save_fewshot_path"])!=True:
        os.mkdir(train_config["save_fewshot_path"])
    if os.path.exists(fewshot_dir)!=True:
        os.mkdir(fewshot_dir)

    np.save(os.path.join(fewshot_dir, "train_index"), trainset)
    np.save(os.path.join(fewshot_dir, "val_index"), valset)
    np.save(os.path.join(fewshot_dir, "test_index"), testset)
    np.save(os.path.join(fewshot_dir, "train_labels"), train_label)
    np.save(os.path.join(fewshot_dir, "val_labels"), val_label)


def split(config):
    for num in trange(config["graph_num"]):
        save_path=os.path.join(config["save_data_dir"],str(num))
        graph=dgl.load_graphs(save_path)[0][0]
        max_nlabel=graph.ndata["label"].max()
        trainset,valset,testset=few_shot_split_nodelevel(graph,config["few_shot_tasknum"],config["train_shotnum"],
                                                         config["val_shotnum"],max_nlabel+1,
                                                         config["seed"],config["split_drop"])
        trainset = np.array(trainset)
        valset = np.array(valset)
        testset = np.array(testset)
        fewshot_dir=os.path.join(config["save_fewshot_dir"],"%s_trainshot_%s_valshot_%s_tasks" %
                                 (config["train_shotnum"],config["val_shotnum"],config["few_shot_tasknum"]))
        if os.path.exists(config["save_fewshot_dir"])!=True:
            os.mkdir(config["save_fewshot_dir"])
        if os.path.exists(fewshot_dir)!=True:
            os.mkdir(fewshot_dir)
        temp=os.path.join(fewshot_dir,str(num))
        if os.path.exists(temp)!=True:
            os.mkdir(os.path.join(fewshot_dir,str(num)))
        np.save(os.path.join(fewshot_dir, str(num),"train_dgl_dataset"), trainset)
        np.save(os.path.join(fewshot_dir, str(num),"val_dgl_dataset"), valset)
        np.save(os.path.join(fewshot_dir, str(num),"test_dgl_dataset"), testset)




