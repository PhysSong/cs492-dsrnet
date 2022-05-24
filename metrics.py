import itertools
import numpy as np
import torch as th
nn = th.nn
F = th.nn.functional

def apply_permutation(pairwise: np.array, permutation):
    """
    Applies permutation to batched pairwise array
    """
    B, M, N = pairwise.shape
    assert M == N
    assert N == len(permutation)
    return np.stack([pairwise[:, i, permutation[i]] for i in range(N)], axis=-1)

def calc_pairwise_inter_union(logit, mask_gt):
    """
    Args
    logit_list: [B, K, S1, S2, S3], softmax, K-1 is empty
    mask_gt_list: [B, S1, S2, S3], 0 is empty
    Returns
    inter: [B, K-1, K-1], voxels in the intersection of ground truth and prediction
    union: [B, K-1, K-1], voxels in the union of ground truth and prediction
    """
    B, K, S1, S2, S3 = logit.size()
    K -= 1
    inter = np.zeros([B, K, K])
    union = np.zeros([B, K, K])
    mask_pred_onehot = F.one_hot(th.argmax(logit, dim=1))[..., :-1]
    mask_gt_onehot = F.one_hot(mask_gt)[..., 1:]
    for b in range(B):
        for i in range(K):
            for j in range(K):
                # TODO find a better way for doing this
                occup_pred = mask_pred_onehot[b, :, :, :, i].to(dtype=th.bool)
                occup_gt = mask_gt_onehot[b, :, :, :, j].to(dtype=th.bool)
                inter[b, i, j] = th.sum(occup_gt * occup_pred).item()
                union[b, i, j] = th.sum(occup_gt + occup_pred).item()

    return inter, union

def calc_iou_for_permutation(inter, union, permutation):
    """
    inter: batched pairwise intersection
    union: batched pairwise union
    """
    inter_permuted = apply_permutation(inter, permutation)
    union_permuted = apply_permutation(union, permutation)
    # TODO how to handle empty union?
    return np.mean(inter_permuted / np.maximum(union_permuted, 1), axis=1)

def calc_iou_from_prediction(logit_list, mask_gt_list, ordered):
    B, K, _, _, _ = logit_list[0].size()
    L = len(logit_list)
    inter_union_list = [
        calc_pairwise_inter_union(logit, mask_gt)
        for logit, mask_gt in zip(logit_list, mask_gt_list)]
    if ordered:
        # Apply the same permutation for entire sequence, and get the best result
        best_iou = np.zeros(B)
        for permutation in itertools.permutations(range(K - 1)):
            frame_iou_sum = np.zeros(B)
            for inter, union in inter_union_list:
                frame_iou_sum += calc_iou_for_permutation(inter, union, permutation)
            best_iou = np.maximum(best_iou, frame_iou_sum / L)
        return best_iou
    else:
        # Try all permutations for each frame, and average the result over sequence
        frame_iou_sum = np.zeros(B)
        for inter, union in inter_union_list:
            best_iou = np.zeros(B)
            for permutation in itertools.permutations(range(K - 1)):
                best_iou = np.maximum(best_iou, calc_iou_for_permutation(inter, union, permutation))
            frame_iou_sum += best_iou
        return frame_iou_sum / L
