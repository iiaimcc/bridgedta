import torch
import random
import numpy as np
from math import sqrt
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def rmse(y, f):
    rmse = sqrt(((y - f) ** 2).mean(axis=0))
    return rmse


def mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse

# from MraphDTA
def get_cindex(gt, pred):
    gt_mask = gt.reshape((1, -1)) > gt.reshape((-1, 1))
    diff = pred.reshape((1, -1)) - pred.reshape((-1, 1))
    h_one = (diff > 0)
    h_half = (diff == 0)
    CI = np.sum(gt_mask * h_one * 1.0 + gt_mask * h_half * 0.5) / np.sum(gt_mask)

    return CI

# from GraphDTA
def get_ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci

# from PocketDTA
def get_aupr(Y, P, threshold=7.0):
    # 7.0 for davis; 12.1 for KIBA
    Y = np.where(Y >= threshold, 1, 0)
    P = np.where(P >= threshold, 1, 0)
    prec, re, _ = precision_recall_curve(Y, P)
    aupr = auc(re, prec)
    return aupr

# not used
def get_auc(Y, P):
    auc = roc_auc_score(Y, P)
    return auc

def get_pcc(X, Y):
    return np.corrcoef(X, Y)[0,1]

def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs

def r_squared_error(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred-y_pred_mean) * (y_obs-y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs-y_obs_mean) * (y_obs-y_obs_mean))
    y_pred_sq = sum((y_pred-y_pred_mean) * (y_pred-y_pred_mean))

    return mult / float(y_obs_sq*y_pred_sq)

def get_k(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs*y_pred) / float(sum(y_pred*y_pred))

def squared_error_zero(y_obs,y_pred):
    k = get_k(y_obs,y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs-(k*y_pred)) * (y_obs-(k*y_pred)))
    down= sum((y_obs-y_obs_mean) * (y_obs-y_obs_mean))

    return 1 - (upp/float(down))

def get_rm2(ys_orig,ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute(r2 - r02)))

def compute_metrics(y_true, y_pred, threshold=7.0):
    """
    计算多个评估指标
    Args:
        y_true: 真实值
        y_pred: 预测值
        threshold: AUPR计算的阈值，默认为7.0 (Davis数据集用7.0，KIBA数据集用12.1)
    Returns:
        包含所有指标的字典
    """
    metrics = {}
    # 原始为list，转换为numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # 计算各项指标
    metrics['ci'] = get_cindex(y_true, y_pred)
    metrics['mse'] = mse(y_true, y_pred)
    metrics['rmse'] = rmse(y_true, y_pred)
    metrics['rm2'] = get_rm2(y_true, y_pred)
    metrics['pearson'] = pearson(y_true, y_pred)
    metrics['spearman'] = spearman(y_true, y_pred)
    metrics['pcc'] = get_pcc(y_true, y_pred)
    metrics['aupr'] = get_aupr(y_true, y_pred, threshold)
    
    return metrics





