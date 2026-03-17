import time
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import json

from methods.GPAWP.utils.pytorchtools import EarlyStopping
from methods.GPAWP.utils.data import load_data
import matplotlib.pyplot as plt
import numpy as np
from purning import calculate_soft_prompt_semantic_importance,calculate_soft_prompt_weight_importance

from GNN import GCN, GAT, GIN,semantic_GCN,myGAT
import dgl
from dgl.nn.pytorch import GraphConv

from prompt import hnode_prompt_layer_feature_weighted_sum,node_prompt_layer_feature_weighted_sum,distance2center,center_embedding\
    ,acm_hnode_prompt_layer_feature_weighted_sum,prompt_gcn,hprompt_gcn\
    ,dblp_hnode_prompt_layer_feature_weighted_sum\
    ,freebase_des_hnode_prompt_layer_feature_weighted_sum,freebase_bidirection_hnode_prompt_layer_feature_weighted_sum\
    ,freebase_source_hnode_prompt_layer_feature_weighted_sum,acm_hnode_semantic_prompt_layer_feature_weighted_sum\
    ,freebase_bidirection_semantic_hnode_prompt_layer_feature_weighted_sum,dblp_hnode_semantic_prompt_layer_feature_weighted_sum\
    ,imdb_hnode_semantic_prompt_layer_feature_weighted_sum


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


def to_list_safe(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def scores_to_mask(scores, method='percentile', param=20, topk=None):
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    scores = np.array(scores).astype(float)

    n = scores.shape[0]
    mask = np.ones(n, dtype=int)

    if method == 'percentile':
        thresh = np.percentile(scores, param)
        mask = (scores > thresh).astype(int)
        if mask.sum() == 0:
            mask[np.argmax(scores)] = 1

    elif method == 'threshold':
        mask = (scores >= param).astype(int)
        if mask.sum() == 0:
            mask[np.argmax(scores)] = 1

    elif method == 'topk':
        idx = np.argsort(-scores)[:topk]
        mask = np.zeros(n, dtype=int)
        mask[idx] = 1

    elif method == 'zscore':
        mu = scores.mean()
        sigma = scores.std() if scores.std() > 0 else 1
        z = (scores - mu) / sigma
        mask = (z >= param).astype(int)
        if mask.sum() == 0:
            mask[np.argmax(scores)] = 1

    return mask


def save_masks(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(data, path)

    json_path = path.replace('.pt', '.json')
    with open(json_path, 'w') as f:
        json.dump({k: to_list_safe(v) for k, v in data.items()}, f, indent=2)


def min_max_normalization(data):
    min_val = torch.min(data)
    max_val = torch.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

def z_score_normalization(data):
    mean = torch.mean(data)
    std_dev = torch.std(data)
    normalized_data = (data - mean) / std_dev
    return normalized_data

def decimal_scaling(data):
    max_abs = torch.max(torch.abs(data))
    k = torch.ceil(torch.log10(max_abs))
    normalized_data = data / (10 ** k)
    return normalized_data

def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)


def subgraph_nodelist(g,subgraphs_dir):
    if os.path.exists(subgraphs_dir) == False:
        os.mkdir(subgraphs_dir)

    if os.path.exists(os.path.join(subgraphs_dir,'0.npy'))==False:
        nodenum = g.number_of_nodes()
        subgraph_list = []
        for i in range(nodenum):
            neighbors = g.successors(i).numpy().tolist()
            two_hop_neighbors = []
            for neighbor in neighbors:
                two_hop_neighbors.extend(g.successors(neighbor).numpy().tolist())
            subgraph_nodes = [i] + neighbors + two_hop_neighbors
            subgraph_nodes = np.array(list(set(subgraph_nodes)))
            subgraph_dir = os.path.join(subgraphs_dir, str(i))
            np.save(subgraph_dir, subgraph_nodes)
            subgraph_list.append(torch.tensor(subgraph_nodes))
    else:
        # Load subgraphs list
        subgraph_list = []
        file_names = [file_name for file_name in os.listdir(subgraphs_dir) if file_name.endswith('.npy')]

        sorted_file_names = sorted(file_names, key=lambda x: int(x.split('.')[0]))

        for file_name in sorted_file_names:
            file_path = os.path.join(subgraphs_dir, file_name)
            np_array = np.load(file_path)
            subgraph_nodes = torch.tensor(np_array)
            subgraph_list.append(subgraph_nodes)
    return subgraph_list




def run_model(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    feats_type = args.feats_type
    index_dir = [str(args.shotnum), 'shots', str(args.tasknum), 'tasks']
    index = "".join(index_dir)
    features_list, adjM, labels, train_val_test_idx, dl = load_data(args.dataset, args.tasknum, args.shotnum, index)


    if args.device == 1:
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    elif args.device == 0:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    features_list = [mat2tensor(features).to(device) for features in features_list]
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        in_dims = []  # [features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2 or feats_type == 4:
        save = feats_type - 2
        in_dims = [features.shape[0] for features in features_list]
        for i in range(0, len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)

    edge2type = {}
    for k in dl.links['data']:
        for u, v in zip(*dl.links['data'][k].nonzero()):
            edge2type[(u, v)] = k
    for i in range(dl.nodes['total']):
        if (i, i) not in edge2type:
            edge2type[(i, i)] = len(dl.links['count'])
    for k in dl.links['data']:
        for u, v in zip(*dl.links['data'][k].nonzero()):
            if (v, u) not in edge2type:
                edge2type[(v, u)] = k + 1 + len(dl.links['count'])
    g = dgl.DGLGraph(adjM + (adjM.T))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)
    trans_g = dgl.reverse(g).to(device)

    coo_adj = adjM.tocoo()
    values = coo_adj.data
    indices = np.vstack((coo_adj.row, coo_adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo_adj.shape

    e_feat = []
    for u, v in zip(*g.edges()):
        u = u.cpu().item()
        v = v.cpu().item()
        e_feat.append(edge2type[(u, v)])
    e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)

    eval_result = {}
    eval_result['micro-f1'] = []
    eval_result['macro-f1'] = []
    train_time = 0
    test_time = 0

    num_classes = dl.labels_train['num_classes']
    if args.model_type == 'gat':
        heads = [args.num_heads] * args.num_layers + [1]
        net = GAT(g, in_dims, args.hidden_dim, num_classes, args.num_layers, heads, F.elu, args.dropout, args.dropout,
                  args.slope, False)
    elif args.model_type == 'gcn':
        if args.pretrain_semantic:
            net = semantic_GCN(g, in_dims, args.hidden_dim, num_classes, args.num_layers, F.elu, args.dropout)
        else:
            net = GCN(g, in_dims, args.hidden_dim, num_classes, args.num_layers, F.elu, args.dropout)
    elif args.model_type == 'gin':
        net = GIN(g, in_dims, args.hidden_dim, num_classes, args.num_layers, F.relu, args.dropout)
    elif args.model_type == 'SHGN':
        num_classes = dl.labels_train['num_classes']
        heads = [args.num_heads] * args.num_layers + [1]
        net = myGAT(g, args.edge_feats, len(dl.links['count']) * 2 + 1, in_dims, args.hidden_dim, num_classes,
                    args.num_layers, heads, F.elu, args.dropout, args.dropout, args.slope, True, 0.05)

    else:
        raise Exception('{} model is not defined!'.format(args.model_type))
    if args.model_type == 'SHGN':
        net.load_state_dict(
            torch.load('./checkpoint/pretrain/shgn_checkpoint_{}_{}.pt'.format(
                args.dataset, args.num_layers)))
    else:
        if args.dataset == 'Freebase':
            if args.load_pretrain:
                if args.hetero_pretrain:
                    if args.hetero_pretrain_subgraph:
                        if args.pretrain_semantic:
                            net.load_state_dict(
                                torch.load(
                                    './checkpoint/pretrain/checkpoint_hsubgraph_semantic_{}_{}_{}_{}_{}_{}_{}.pt'.
                                    format(args.dataset, args.model_type, args.subgraph_hop_num,
                                           args.pre_loss_weight, args.feats_type, args.tuple_neg_disconnected_num,
                                           args.tuple_neg_unrelated_num)))
                        elif args.pretrain_each_loss:
                            net.load_state_dict(
                                torch.load(
                                    './checkpoint/pretrain/checkpoint_hsubgraph_each_loss_{}_{}_{}_{}_{}_{}_{}.pt'.
                                    format(args.dataset, args.model_type, args.subgraph_hop_num,
                                           args.pre_loss_weight, args.feats_type, args.tuple_neg_disconnected_num,
                                           args.tuple_neg_unrelated_num)))

                        else:
                            net.load_state_dict(
                                torch.load('./checkpoint/pretrain/checkpoint_hsubgraph_{}_{}_{}_{}_{}_{}_{}.pt'.
                                           format(args.dataset, args.model_type, args.subgraph_hop_num,
                                                  args.pre_loss_weight, args.feats_type,
                                                  args.tuple_neg_disconnected_num, args.tuple_neg_unrelated_num)))
                    else:
                        net.load_state_dict(
                            torch.load(
                                './checkpoint/pretrain/checkpoint_{}_{}_{}_{}_{}_{}_{}.pt'.
                                format(args.dataset, args.model_type, args.subgraph_hop_num,
                                       args.pre_loss_weight, args.feats_type, args.tuple_neg_disconnected_num,
                                       args.tuple_neg_unrelated_num)))
                else:
                    if args.hetero_pretrain_subgraph:
                        if args.pretrain_semantic:
                            net.load_state_dict(
                                torch.load('./checkpoint/pretrain/checkpoint_hsubgraph_semantic_{}_{}_{}_{}_{}_{}.pt'.
                                           format(args.dataset, args.model_type, args.subgraph_hop_num,
                                                  args.feats_type, args.tuple_neg_disconnected_num,
                                                  args.tuple_neg_unrelated_num)))
                        elif args.pretrain_each_loss:
                            net.load_state_dict(
                                torch.load(
                                    './checkpoint/pretrain/checkpoint_hsubgraph_each_loss_{}_{}_{}_{}_{}_{}_{}.pt'.
                                    format(args.dataset, args.model_type, args.subgraph_hop_num,
                                           args.pre_loss_weight, args.feats_type, args.tuple_neg_disconnected_num,
                                           args.tuple_neg_unrelated_num)))
                        else:
                            net.load_state_dict(
                                torch.load('./checkpoint/pretrain/checkpoint_hsubgraph_{}_{}_{}_{}_{}_{}.pt'.
                                           format(args.dataset, args.model_type, args.subgraph_hop_num,
                                                  args.feats_type, args.tuple_neg_disconnected_num,
                                                  args.tuple_neg_unrelated_num)))
                    else:
                        net.load_state_dict(
                            torch.load(
                                './checkpoint/pretrain/checkpoint_{}_{}_{}_{}_{}_{}.pt'.
                                format(args.dataset, args.model_type, args.subgraph_hop_num,
                                       args.feats_type, args.tuple_neg_disconnected_num, args.tuple_neg_unrelated_num)))
        ##################### end of if freebase#############
        else:
            if args.load_pretrain:
                if args.hetero_pretrain:
                    if args.hetero_pretrain_subgraph:
                        if args.pretrain_semantic:
                            net.load_state_dict(
                                torch.load('./checkpoint/pretrain/checkpoint_hsubgraph_semantic_{}_{}_{}_{}_{}.pt'.
                                           format(args.dataset, args.model_type, args.subgraph_hop_num,
                                                  args.pre_loss_weight, args.feats_type)))
                        elif args.pretrain_each_loss:
                            net.load_state_dict(
                                torch.load('./checkpoint/pretrain/checkpoint_hsubgraph_each_loss_{}_{}_{}_{}_{}.pt'.
                                           format(args.dataset, args.model_type, args.subgraph_hop_num,
                                                  args.pre_loss_weight, args.feats_type)))

                        else:
                            net.load_state_dict(
                                torch.load('./checkpoint/pretrain/checkpoint_hsubgraph_{}_{}_{}_{}_{}.pt'.
                                           format(args.dataset, args.model_type, args.subgraph_hop_num,
                                                  args.pre_loss_weight, args.feats_type)))
                    else:
                        net.load_state_dict(
                            torch.load(
                                './checkpoint/pretrain/checkpoint_{}_{}_{}_{}_{}.pt'.
                                format(args.dataset, args.model_type, args.subgraph_hop_num, args.pre_loss_weight,
                                       args.feats_type)))
                else:
                    if args.hetero_pretrain_subgraph:
                        if args.pretrain_semantic:
                            net.load_state_dict(
                                torch.load('./checkpoint/pretrain/checkpoint_hsubgraph_semantic_{}_{}_{}_{}.pt'.
                                           format(args.dataset, args.model_type, args.subgraph_hop_num,
                                                  args.feats_type)))
                        elif args.pretrain_each_loss:
                            net.load_state_dict(
                                torch.load('./checkpoint/pretrain/checkpoint_hsubgraph_each_loss_{}_{}_{}_{}_{}.pt'.
                                           format(args.dataset, args.model_type, args.subgraph_hop_num,
                                                  args.pre_loss_weight, args.feats_type)))
                        else:
                            net.load_state_dict(
                                torch.load('./checkpoint/pretrain/checkpoint_hsubgraph_{}_{}_{}_{}.pt'.
                                           format(args.dataset, args.model_type, args.subgraph_hop_num,
                                                  args.feats_type)))
                    else:
                        net.load_state_dict(
                            torch.load(
                                './checkpoint/pretrain/checkpoint_{}_{}_{}_{}.pt'.
                                format(args.dataset, args.model_type, args.subgraph_hop_num, args.feats_type)))

    net.to(device)
    if args.pretrain_semantic:
        prelogits, semantic_weight = net(features_list)
    else:
        if args.model_type == 'SHGN':
            prelogits = net(features_list, e_feat)
        else:
            prelogits = net(features_list)

    prelogits =prelogits.to(device)


    if args.tuning == 'linear':
        classify = torch.nn.Linear(args.hidden_dim, num_classes)
    elif args.tuning == 'gcn':
        classify = GraphConv(args.hidden_dim, num_classes)
    elif args.tuning in ('weight-sum', 'weight-sum-center-fixed', 'bottle-net'):
        if args.model_type == 'SHGN':
            hidden_dim = args.shgn_hidden_dim
        else:
            hidden_dim = args.hidden_dim
        if args.add_edge_info2prompt:
            classify = hnode_prompt_layer_feature_weighted_sum(hidden_dim)
            if args.each_type_subgraph:
                if args.dataset == 'ACM':
                    if args.pretrain_semantic:
                        classify = acm_hnode_prompt_layer_feature_weighted_sum(hidden_dim, semantic_weight)
                    elif args.pretrain_each_loss:
                        classify = acm_eachloss_hnode_prompt_layer_feature_weighted_sum(hidden_dim,
                                                                                        semantic_weight)
                    elif args.semantic_prompt == 1:
                        classify = acm_hnode_semantic_prompt_layer_feature_weighted_sum(hidden_dim,
                                                                                        semantic_prompt_weight=args.semantic_prompt_weight)
                    else:
                        classify = acm_hnode_prompt_layer_feature_weighted_sum(hidden_dim)
                elif args.dataset == 'DBLP':
                    if args.semantic_prompt == 1:
                        classify = dblp_hnode_semantic_prompt_layer_feature_weighted_sum(hidden_dim,
                                                                                         semantic_prompt_weight=args.semantic_prompt_weight)
                    else:
                        classify = dblp_hnode_prompt_layer_feature_weighted_sum(hidden_dim)
                elif args.dataset == 'Freebase':
                    if args.semantic_prompt == 1:
                        classify = freebase_bidirection_semantic_hnode_prompt_layer_feature_weighted_sum(
                            hidden_dim, semantic_prompt_weight=args.semantic_prompt_weight)
                    else:
                        if args.freebase_type == 2:
                            classify = freebase_bidirection_hnode_prompt_layer_feature_weighted_sum(hidden_dim)
                        elif args.freebase_type == 1:
                            classify = freebase_des_hnode_prompt_layer_feature_weighted_sum(hidden_dim)
                        else:
                            classify = freebase_source_hnode_prompt_layer_feature_weighted_sum(hidden_dim)
                elif args.dataset == 'IMDB':
                    classify = imdb_hnode_semantic_prompt_layer_feature_weighted_sum(hidden_dim,
                                                                                        semantic_prompt_weight=args.semantic_prompt_weight)


        else:
            if args.tuning == 'bottle-net':
                print("##############    bottel-net   ###############")
                classify = node_bottle_net(args.hidden_dim, args.bottle_net_hidden_dim,
                                           args.bottle_net_output_dim)
            else:
                classify = node_prompt_layer_feature_weighted_sum(hidden_dim)
    elif args.tuning in ('cat'):
        classify = node_prompt_layer_feature_cat(args.cat_prompt_dim)
    elif args.tuning in ('sum'):
        if args.add_edge_info2prompt:
            classify = hnode_prompt_layer_feature_sum()
        else:
            classify = node_prompt_layer_feature_sum()
    elif args.tuning in ('cat_edge'):
        if args.add_edge_info2prompt:
            classify = hnode_prompt_layer_feature_cat_edge(args.cat_prompt_dim, args.cat_hprompt_dim)
        else:
            classify = node_prompt_layer_feature_cat_edge(args.cat_prompt_dim)
    elif args.tuning in ('prompt_gcn'):
        if args.add_edge_info2prompt:
            classify = hprompt_gcn(args.hidden_dim)
        else:
            classify = prompt_gcn(args.hidden_dim)

    else:
        print('tuning model does not exist')
        sys.exit()

    classify.to(device)

    if args.tuning != 'sum':
        optimizer = torch.optim.AdamW(classify.parameters(),
                                      lr=args.lr, weight_decay=args.weight_decay)

    # training loop
    classify.train()
    early_stopping_classify = EarlyStopping(patience=args.patience, verbose=True,
                                            save_path='./checkpoint/retrain/checkpoint_{}_{}_{}_freeze_classify.pt'.format(
                                                args.dataset, args.model_type, args.tuning, args.shotnum))

    classify.load_state_dict(torch.load(
        './checkpoint/checkpoint_{}_{}_{}_{}_freeze_classify.pt'.format(
            args.dataset, args.model_type, args.tuning, args.shotnum)))
    if args.dataset=="Freebase":
        token_mask = 1 - torch.eye(9, dtype=torch.int64)
    else:
        token_mask = 1 - torch.eye(5, dtype=torch.int64)

    num_classes = dl.labels_train['num_classes']

    semantic_token_importance, softmax_semantic_importance = \
        calculate_soft_prompt_semantic_importance(
            classify,
            token_mask,
            g,
            prelogits,
            e_feat,
            train_val_test_idx['val_idx'],
            labels['val'],
            num_classes,
            args.device,
            trans_g,
            args.dataset
        )


    weight_masks = []
    for i in range(args.num_subweight):
        mask = [1] * args.hidden_dim
        start_index = i * (args.hidden_dim // args.num_subweight)
        for j in range(args.hidden_dim // args.num_subweight):
            mask[start_index + j] = 0
        weight_masks.append(mask)

    weight_mask = torch.tensor(weight_masks)

    weight_token_importance, softmax_weight_importance = \
        calculate_soft_prompt_weight_importance(
            classify,
            weight_mask,
            g,
            prelogits,
            e_feat,
            train_val_test_idx['val_idx'],
            labels['val'],
            num_classes,
            args.num_subweight,
            args.device,
            trans_g,
            args.dataset
        )

    print("P_S importance scores", semantic_token_importance, softmax_semantic_importance)
    print("P_F importance scores", weight_token_importance, softmax_weight_importance)


    sem_mask = scores_to_mask(
        semantic_token_importance,
        method=args.mask_method_semantic,
        param=args.mask_param_semantic
    )

    weight_submask = scores_to_mask(
        weight_token_importance,
        method=args.mask_method_weight,
        param=args.mask_param_weight
    )

    sub_dim = args.hidden_dim // args.num_subweight
    weight_full = []

    for i, keep in enumerate(weight_submask):
        if keep == 1:
            weight_full += [1] * sub_dim
        else:
            weight_full += [0] * sub_dim
    weight_full = np.array(weight_full)

    save_dict = {
        'semantic_mask': to_numpy(sem_mask),
        'weight_submask': to_numpy(weight_submask),
        'weight_full': to_numpy(weight_full),
        'semantic_scores': to_numpy(semantic_token_importance),
        'weight_scores': to_numpy(weight_token_importance)
    }

    save_path = './checkpoint/best_mask.pt'
    save_masks(save_path, save_dict)
    print("Saved mask to:", save_path)


    with open("./checkpoint/best_mask.txt", "a") as f:
        print("\n", file=f)
        print("dataset", args.dataset, file=f)
        print("semantic mask", sem_mask, file=f)
        print("weight mask", weight_submask, file=f)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset')
    ap.add_argument('--feats-type', type=int, default=2,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2;' +
                        '4 - only term features (id vec for others);' +
                        '5 - only term features (zero vec for others).')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--bottle-net-hidden-dim', type=int, default=2, help='Dimension of the node hidden state. Default is 2.')
    ap.add_argument('--bottle-net-output-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--edge-feats', type=int, default=64)
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--epoch', type=int, default=1, help='Number of epochs.')
    ap.add_argument('--patience', type=int, default=1, help='Patience.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--model-type', type=str, default='gcn', help="gcn or gat")
    ap.add_argument('--num-layers', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--run', type=int, default=1)
    ap.add_argument('--device',type=int,default=1)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--weight-decay', type=float, default=1e-6)
    ap.add_argument('--slope', type=float, default=0.05)
    ap.add_argument('--dataset', type=str, default='ACM')
    ap.add_argument('--seed', type=int, default="0")
    ap.add_argument('--tasknum', type=int, default=100)
    ap.add_argument('--shotnum', type=int, default=1)
    ap.add_argument('--load_pretrain', type=int, default=1)
    ap.add_argument('--tuning',type=str, default='weight-sum-center-fixed')
    ap.add_argument('--subgraph_hop_num', type=int, default=1)
    #make sure we have ran pre_train_model with this loss_weight
    ap.add_argument('--pre_loss_weight', type=float, default=1)
    ap.add_argument('--hetero_pretrain', type=int, default=0)
    ap.add_argument('--hetero_pretrain_subgraph', type=int, default=0)
    ap.add_argument('--pretrain_semantic', type=int, default=0)
    ap.add_argument('--add_edge_info2prompt', type=int, default=1)
    ap.add_argument('--each_type_subgraph', type=int, default=1)
    ap.add_argument('--pretrain_each_loss', type=int, default=0)
    ap.add_argument('--cat_prompt_dim', type=int, default=64, help='Dimension of the cat prompt dim. Default is 64.')
    ap.add_argument('--cat_hprompt_dim', type=int, default=64, help='Dimension of the cat prompt dim. Default is 64.')
    ap.add_argument('--tuple_neg_disconnected_num', type=int, default=1, help='Dimension of the cat prompt dim. Default is 64.')
    ap.add_argument('--tuple_neg_unrelated_num', type=int, default=1, help='Dimension of the cat prompt dim. Default is 64.')
    ap.add_argument('--meta_path', type=int, default=0)
    ap.add_argument('--semantic-prompt', type=int, default=1)
    ap.add_argument('--freebase-type', type=int, default=0, help='0:book as source, 1:book as destination, 2: bidirection')
    ap.add_argument('--semantic-prompt-weight', type=float, default=0.1)
    ap.add_argument('--shgn-hidden-dim', type=int, default=3, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num_subweight', type=int, default=16, help='number of weight tokens')
    ap.add_argument('--use_best_mask', action='store_true', help='apply auto masks in retrain')
    ap.add_argument('--mask_method_semantic', default='percentile')
    ap.add_argument('--mask_param_semantic', type=float, default=20)
    ap.add_argument('--mask_method_weight', default='percentile')
    ap.add_argument('--mask_param_weight', type=float, default=20)

    args = ap.parse_args()
    run_model(args)
