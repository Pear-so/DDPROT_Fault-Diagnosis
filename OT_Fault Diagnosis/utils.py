#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import pandas as pd
import numpy as np
import csv
from sklearn.metrics import confusion_matrix
from ot.utils import unif, dist, list_to_array
from ot.backend import get_backend
import warnings
from scipy.optimize.linesearch import scalar_search_armijo
import os.path as osp
import os

# 默认参数设置
DEFAULT_NUM_ITERMAX = 1000
DEFAULT_STOP_THR = 1e-9

def computeTransportSinkhorn(distributS, distributT, cost_matrix, reg, 
                            numItermax=DEFAULT_NUM_ITERMAX, 
                            stopThr=DEFAULT_STOP_THR):
    """
    使用Sinkhorn算法计算两个分布之间的最优传输矩阵
    
    参数:
        distributS: 源域分布 (np.ndarray)
        distributT: 目标域分布 (np.ndarray)
        cost_matrix: 代价矩阵 (np.ndarray)
        reg: 熵正则化参数 (float)
        numItermax: 最大迭代次数 (int)
        stopThr: 收敛阈值 (float)
        
    返回:
        最优传输矩阵 (np.ndarray)
    """
    dim_S = len(distributS)
    dim_T = len(distributT)
    
    # 构建Gibbs核
    K = np.exp(cost_matrix / -reg)
    Kp = np.dot(np.diag(1/distributS), K)
    
    # 初始化迭代变量
    u = np.ones(dim_S) / dim_S
    v = np.ones(dim_T) / dim_T
    
    err = 1
    for i in range(numItermax):
        uprev = u
        vprev = v
        
        # Sinkhorn迭代
        KtransposeU = np.dot(K.T, u)
        v = distributT / KtransposeU
        u = 1. / np.dot(Kp, v)
        
        # 检查数值稳定性
        if (np.any(KtransposeU == 0) or 
            np.any(np.isnan(u)) or np.any(np.isnan(v)) or
            np.any(np.isinf(u)) or np.any(np.isinf(v))):
            warnings.warn(f'数值不稳定警告: 迭代 {i} 出现数值错误')
            u = uprev
            v = vprev
            break
            
        # 每10次迭代检查收敛性
        if i % 10 == 0:
            transp = np.dot(np.diag(u), np.dot(K, np.diag(v)))
            err = np.linalg.norm(np.sum(transp, axis=0) - distributT)**2
            if err < stopThr:
                break
                
    return np.dot(np.diag(u), np.dot(K, np.diag(v)))

def gcg(distributS, distributT, cost_matrix, reg1, reg2, f, df, G0=None,
        numItermax=100, numInnermax=100, stopThr=1e-9, stopThr2=1e-9):
    """
    使用梯度条件梯度(GCG)算法求解带约束的最优传输问题
    
    参数:
        distributS: 源域分布 (np.ndarray)
        distributT: 目标域分布 (np.ndarray)
        cost_matrix: 代价矩阵 (np.ndarray)
        reg1: 熵正则化参数 (float)
        reg2: 函数f的正则化参数 (float)
        f: 正则化函数
        df: f的梯度函数
        G0: 初始传输矩阵 (np.ndarray, 可选)
        numItermax: 外层最大迭代次数 (int)
        numInnermax: 内层最大迭代次数 (int)
        stopThr: 相对收敛阈值 (float)
        stopThr2: 绝对收敛阈值 (float)
        
    返回:
        最优传输矩阵 (np.ndarray)
    """
    # 初始化传输矩阵
    if G0 is None:
        G = np.outer(distributS, distributT)
    else:
        G = G0
        
    # 定义总代价函数
    def cost(G):
        """计算带熵正则化和函数f正则化的总代价"""
        return np.sum(cost_matrix * G) + reg1 * np.sum(G * np.log(G)) + reg2 * f(G)
    
    f_val = cost(G)
    loop = 1
    it = 0
    
    # 主迭代循环
    while loop:
        it += 1
        old_fval = f_val
        
        # 计算当前梯度并调整代价矩阵
        Mi = cost_matrix + reg2 * df(G)
        if np.any(Mi < 0):
            Mi += -np.min(Mi) + 1e-6  # 确保所有元素非负
            
        # 使用Sinkhorn算法求解子问题
        Gc = computeTransportSinkhorn(distributS, distributT, Mi, reg1, 
                                     numItermax=numInnermax, stopThr=1e-9)
        deltaG = Gc - G
        
        # 计算方向导数
        dcost = Mi + reg1 * (1 + np.log(G))
        
        # 使用Armijo线搜索确定步长
        alpha, fc, f_val = line_search_armijo(
            cost, G, deltaG, dcost, old_fval, alpha_min=0., alpha_max=1.)
        
        # 更新传输矩阵
        G = G + alpha * deltaG
        
        # 检查收敛条件
        if it >= numItermax:
            loop = 0
            
        abs_delta_fval = abs(f_val - old_fval)
        relative_delta_fval = abs_delta_fval / abs(f_val)
        
        if relative_delta_fval < stopThr or abs_delta_fval < stopThr2:
            loop = 0
            
    return G

def line_search_armijo(f, xk, pk, gfk, old_fval, args=(), c1=1e-4,
                       alpha0=0.99, alpha_min=0, alpha_max=1):
    """
    使用Armijo条件进行线搜索确定最优步长
    
    参数:
        f: 目标函数
        xk: 当前点
        pk: 搜索方向
        gfk: 梯度
        old_fval: 旧的函数值
        args: 额外参数
        c1: Armijo条件参数
        alpha0: 初始步长
        alpha_min: 最小步长
        alpha_max: 最大步长
        
    返回:
        alpha: 最优步长
        fc: 函数调用次数
        phi1: 新的函数值
    """
    fc = [0]
    
    def phi(alpha1):
        """步长为alpha1时的函数值"""
        fc[0] += 1
        return f(xk + alpha1 * pk, *args)
    
    # 计算初始函数值
    if old_fval is None:
        phi0 = phi(0.)
    else:
        phi0 = old_fval
        
    # 计算方向导数
    derphi0 = np.sum(pk * gfk)
    
    # 使用scipy的Armijo线搜索
    alpha, phi1 = scalar_search_armijo(
        phi, phi0, derphi0, c1=c1, alpha0=alpha0)
        
    # 确保步长在合理范围内
    if alpha is None:
        return 0., fc[0], phi0
    else:
        if alpha_min is not None or alpha_max is not None:
            alpha = np.clip(alpha, alpha_min, alpha_max)
        return float(alpha), fc[0], phi1

def sinkhorn_R1reg_lab(a, b, cost_matrix, reg, eta, numItermax=100,
                     numInnerItermax=100, stopInnerThr=1e-9, 
                     intra_class=None, inter_class=None, aux=None, aux1=None):
    """
    使用带标签信息的R1正则化Sinkhorn算法计算最优传输
    
    参数:
        a: 源域分布 (np.ndarray)
        b: 目标域分布 (np.ndarray)
        cost_matrix: 代价矩阵 (np.ndarray)
        reg: 熵正则化参数 (float)
        eta: R1正则化参数 (float)
        numItermax: 外层最大迭代次数 (int)
        numInnerItermax: 内层最大迭代次数 (int)
        stopInnerThr: 内层收敛阈值 (float)
        intra_class: 类内参数 (torch.Tensor)
        inter_class: 类间参数 (torch.Tensor)
        aux: 辅助矩阵1 (np.ndarray)
        aux1: 辅助矩阵2 (np.ndarray)
        
    返回:
        最优传输矩阵 (np.ndarray)
    """
    # 转换为numpy数组
    intra_class = intra_class.detach().cpu().numpy()
    inter_class = inter_class.detach().cpu().numpy()
    
    # 构建类内和类间约束矩阵
    Intra = np.ones((len(a), len(b))) * intra_class * aux
    Inter = np.ones((len(a), len(b))) * inter_class * aux1
    zero = np.zeros_like(cost_matrix)

    def f(G):
        """R1正则化函数"""
        phi = (G - Inter) * (Intra - G)
        phi = np.where(phi > 0, phi, zero)
        return phi.sum()
    
    def df(G):
        """R1正则化函数的梯度"""
        d_phi = Inter + Intra - 2 * G
        phi = (G - Inter) * (Intra - G)
        return np.where(phi < 0, zero, d_phi)
  
    # 使用GCG算法求解
    return gcg(a, b, cost_matrix, reg, eta, f, df, G0=None, 
               numItermax=numItermax, numInnermax=numInnerItermax, 
               stopThr=stopInnerThr)

def sinkhorn_R1reg(a, b, cost_matrix, reg, eta=0.1, numItermax=10,
                     numInnerItermax=10, stopInnerThr=1e-9, 
                     intra_class=None, inter_class=None):
    """
    使用R1正则化的Sinkhorn算法计算最优传输
    
    参数:
        a: 源域分布 (np.ndarray)
        b: 目标域分布 (np.ndarray)
        cost_matrix: 代价矩阵 (np.ndarray)
        reg: 熵正则化参数 (float)
        eta: R1正则化参数 (float)
        numItermax: 外层最大迭代次数 (int)
        numInnerItermax: 内层最大迭代次数 (int)
        stopInnerThr: 内层收敛阈值 (float)
        intra_class: 类内参数 (torch.Tensor)
        inter_class: 类间参数 (torch.Tensor)
        
    返回:
        最优传输矩阵 (np.ndarray)
    """
    # 转换为numpy数组
    intra_class = intra_class.detach().cpu().numpy()
    inter_class = inter_class.detach().cpu().numpy()
    
    # 构建类内和类间约束矩阵
    Intra = np.ones((len(a), len(b))) * intra_class 
    Inter = np.ones((len(a), len(b))) * inter_class 
    zero = np.zeros_like(cost_matrix)

    def f(G):
        """R1正则化函数"""
        phi = (G - Inter) * (Intra - G)
        phi = np.where(phi > 0, phi, zero)
        return phi.sum()
    
    def df(G):
        """R1正则化函数的梯度"""
        d_phi = Inter + Intra - 2 * G
        phi = (G - Inter) * (Intra - G)
        return np.where(phi < 0, zero, d_phi)

    # 使用GCG算法求解
    return gcg(a, b, cost_matrix, reg, eta, f, df, G0=None, 
               numItermax=numItermax, numInnermax=numInnerItermax, 
               stopThr=stopInnerThr)

def sinkhorn(a, b, cost_matrix, reg, eta=0.1, numItermax=10,
                     numInnerItermax=10, stopInnerThr=1e-9):
    """
    基本Sinkhorn算法的变体实现
    
    参数:
        a: 源域分布 (np.ndarray)
        b: 目标域分布 (np.ndarray)
        cost_matrix: 代价矩阵 (np.ndarray)
        reg: 熵正则化参数 (float)
        eta: 额外正则化参数 (float)
        numItermax: 外层最大迭代次数 (int)
        numInnerItermax: 内层最大迭代次数 (int)
        stopInnerThr: 内层收敛阈值 (float)
        
    返回:
        最优传输矩阵 (np.ndarray)
    """
    zero = np.zeros_like(cost_matrix)

    def f(G):
        """正则化函数"""
        phi = G * (-G)
        phi = np.where(phi > 0, phi, zero)
        return phi.sum()
    
    def df(G):
        """正则化函数的梯度"""
        d_phi = -2 * G
        phi = G * (-G)
        return np.where(phi < 0, zero, d_phi)

    # 使用GCG算法求解
    return gcg(a, b, cost_matrix, reg, eta, f, df, G0=None, 
               numItermax=numItermax, numInnermax=numInnerItermax, 
               stopThr=stopInnerThr)


