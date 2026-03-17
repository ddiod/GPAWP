
import itertools
from prompt import distance2center,center_embedding

import torch
import torch.nn.functional as F
import numpy as np

# Min-max normalization to [0,1]
def normalize_importance(token_importance):
    min_val = token_importance.min()
    max_val = token_importance.max()
    if max_val - min_val > 1e-8:
        norm_importance = (token_importance - min_val) / (max_val - min_val)
    else:
        norm_importance = torch.zeros_like(token_importance)
    return norm_importance


def calculate_soft_prompt_weight_importance(
        classify, mask, g, prelogits, e_feat, val_idx, val_labels,
        num_classes, num_subweight, device, trans_g, dataset, num_tasks=100,
        center_embedding_path_template='./checkpoint/{}_task_center_embedding.npy'
):

    if isinstance(val_labels, np.ndarray):
        val_labels = torch.tensor(val_labels, device=device)
    else:
        val_labels = val_labels.to(device)

    if isinstance(val_idx, np.ndarray):
        val_idx = torch.tensor(val_idx, dtype=torch.long, device=device)
    else:
        val_idx = val_idx.to(device)

    num_blocks = len(mask)
    block_importance = torch.zeros(num_blocks, device=device)

    for i, input_mask in enumerate(mask):
        classify.zero_grad()
        original_weight = classify.save_weight()
        classify.mask_weight_token(input_mask.to(device))
        classify.weight.requires_grad_()
        classify.semantic_weight.requires_grad_()

        if dataset != 'Freebase':
            logits = classify(g, prelogits, e_feat)
        else:
            logits = classify(g, trans_g, prelogits, e_feat)

        for count in range(num_tasks):
            val_label = val_labels[count]
            embedding = logits[val_idx][count]
            c_emb_path = center_embedding_path_template.format(count)
            c_emb = torch.tensor(np.load(c_emb_path), device=device)

            distance = distance2center(embedding, c_emb)
            loss = F.nll_loss(F.log_softmax(distance, dim=1), val_label)
            loss.backward(retain_graph=True)

            grad = classify.weight.grad.to(device)
            block_importance[i] += torch.abs(grad).mean()

        classify.weight_return(original_weight)

    avg_importance = block_importance / num_tasks

    epsilon = 1e-2
    alpha = 1.05
    norm_temp = avg_importance - avg_importance.min()
    norm_temp = norm_temp / (norm_temp.max() + 1e-8)
    normalized = (norm_temp + epsilon) ** alpha
    normalized = normalized / normalized.max()


    return avg_importance, normalized




def calculate_soft_prompt_semantic_importance(
        net, mask, g, prelogits, e_feat, val_idx, val_labels,
        num_classes, device, trans_g, dataset, num_tasks=100,
        center_embedding_path_template='./checkpoint/{}_task_center_embedding.npy'
):

    if isinstance(val_labels, np.ndarray):
        val_labels = torch.tensor(val_labels, device=device)
    else:
        val_labels = val_labels.to(device)

    if isinstance(val_idx, np.ndarray):
        val_idx = torch.tensor(val_idx, dtype=torch.long, device=device)
    else:
        val_idx = val_idx.to(device)

    num_tokens = len(mask)
    token_importance = torch.zeros(num_tokens, device=device)

    for i, input_mask in enumerate(mask):
        net.zero_grad()
        original_weight = net.save_semantic_weight()
        net.mask_semantic_token(input_mask.to(device))
        net.weight.requires_grad_()
        net.semantic_weight.requires_grad_()

        if dataset != 'Freebase':
            logits = net(g, prelogits, e_feat)
        else:
            logits = net(g, trans_g, prelogits, e_feat)

        for count in range(num_tasks):
            val_label = val_labels[count]
            embedding = logits[val_idx][count]
            c_emb_path = center_embedding_path_template.format(count)
            c_emb = torch.tensor(np.load(c_emb_path), device=device)

            distance = distance2center(embedding, c_emb)
            loss = F.nll_loss(F.log_softmax(distance, dim=1), val_label)
            loss.backward(retain_graph=True)

            grad = net.semantic_weight.grad.to(device)
            token_importance[i] += torch.abs(grad).mean()

        net.semantic_weight_return(original_weight)

    avg_importance = token_importance / num_tasks
    normalized = avg_importance / avg_importance.max()
    return avg_importance, normalized




def generate_all_mask_combinations_reverse(num_tokens):
    all_combinations = []
    for r in range(num_tokens, 0, -1):
        combinations = list(itertools.combinations(range(num_tokens), r))
        all_combinations.extend(combinations)
    return all_combinations

def generate_all_mask_combinations(num_tokens):
    all_combinations = []
    for r in range(num_tokens + 1):
        combinations = list(itertools.combinations(range(num_tokens), r))
        all_combinations.extend(combinations)
    return all_combinations



def z_score_normalization(data):
    mean = torch.mean(data)
    std_dev = torch.std(data)
    normalized_data = (data - mean) / std_dev
    return normalized_data