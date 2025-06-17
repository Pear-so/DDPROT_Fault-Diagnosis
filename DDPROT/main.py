# -*- coding: utf-8 -*-
import os, ot
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.weight_norm as weightNorm
import torch.utils.data
from utils import computeTransportSinkhorn, sinkhorn_R1reg, sinkhorn_R1reg_lab
from PU_data import PU_dataset_read
import warnings
from resnet18_1d import resnet18_features
from datetime import datetime
from typing import Tuple, List, Dict, Optional

# 设置计算设备（优先使用GPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_euclidean_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    计算两个特征矩阵之间的欧氏距离矩阵
    
    参数:
        X: 源特征矩阵 (batch_size_x, feature_dim)
        Y: 目标特征矩阵 (batch_size_y, feature_dim)
    
    返回:
        距离矩阵 (batch_size_x, batch_size_y)
    """
    xx = X.pow(2).sum(1).repeat(Y.shape[0], 1)
    xy = X @ Y.t()
    yy = Y.pow(2).sum(1).repeat(X.shape[0], 1)
    dist = xx.t() + yy - 2 * xy
    return dist

def calculate_normalized_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    计算归一化后的欧氏距离矩阵（先对特征向量进行L2归一化）
    
    参数:
        X: 源特征矩阵 (batch_size_x, feature_dim)
        Y: 目标特征矩阵 (batch_size_y, feature_dim)
    
    返回:
        归一化距离矩阵 (batch_size_x, batch_size_y)
    """
    # 计算L2范数并归一化特征向量
    norm_X = X.pow(2).sum(1).pow(0.5).detach().unsqueeze(1)
    norm_Y = Y.pow(2).sum(1).pow(0.5).detach().unsqueeze(1)
    
    X_normalized = X * (1/norm_X)
    Y_normalized = Y * (1/norm_Y)
    
    # 计算归一化后的距离
    xx = X_normalized.pow(2).sum(1).repeat(Y.shape[0], 1)
    xy = X_normalized @ Y_normalized.t()
    yy = Y_normalized.pow(2).sum(1).repeat(X.shape[0], 1)
    dist = xx.t() + yy - 2 * xy
    return dist

def calculate_accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> List[float]:
    """
    计算分类准确率（支持top-k评估）
    
    参数:
        output: 模型输出logits (batch_size, num_classes)
        target: 真实标签 (batch_size,)
        topk: 计算准确率的k值元组
    
    返回:
        各top-k准确率的列表（百分比）
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        # 获取top-k预测结果
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        # 计算正确预测
        correct = pred.eq(target[None])
        accuracy_results = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            accuracy_results.append(correct_k * (100.0 / batch_size))
        return accuracy_results

def initialize_model_weights(m: nn.Module) -> None:
    """
    初始化神经网络层的权重（根据层类型采用不同初始化策略）
    
    参数:
        m: 神经网络层
    """
    layer_type = m.__class__.__name__
    if 'Conv1d' in layer_type or 'ConvTranspose1d' in layer_type:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif 'BatchNorm' in layer_type:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif 'Linear' in layer_type:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class FeatureClassifier(nn.Module):
    """特征分类器（支持权重归一化）"""
    
    def __init__(self, class_num: int = 31, bottleneck_dim: int = 256, norm_type: str = "linear"):
        """
        初始化分类器
        
        参数:
            class_num: 分类类别数
            bottleneck_dim: 特征瓶颈维度
            norm_type: 线性层类型（'linear'或'wn'权重归一化）
        """
        super(FeatureClassifier, self).__init__()
        self.norm_type = norm_type
        if norm_type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(initialize_model_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(initialize_model_weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.fc(x)

def compute_classification_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """计算分类准确率"""
    pred = torch.argmax(logits, dim=1)
    return (pred == labels).type(torch.FloatTensor).mean().item()

def set_gradient_status(model: nn.Module, requires_grad: bool = True) -> None:
    """设置模型参数的梯度更新状态"""
    for param in model.parameters():
        param.requires_grad = requires_grad

def evaluate_model(source_domain: str, target_domain: str, 
                  feature_extractor: nn.Module, classifier: nn.Module,
                  data_loader: torch.utils.data.DataLoader,
                  domain: Optional[str] = None) -> float:
    """
    评估模型在目标域上的分类性能
    
    参数:
        source_domain: 源域标识
        target_domain: 目标域标识
        feature_extractor: 特征提取网络
        classifier: 分类器网络
        data_loader: 数据加载器
        domain: 评估域类型（打印结果时使用）
    
    返回:
        分类准确率（百分比）
    """
    feature_extractor.eval()
    classifier.eval()
    correct_predictions = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images, targets = images.to(device), targets.to(device)
            features = feature_extractor(images)
            outputs = classifier(features)
            pred = outputs.data.max(1)[1]
            correct_predictions += pred.eq(targets.data).cpu().sum()
        accuracy = correct_predictions.item() / len(data_loader.dataset)
    if domain == 'target':
        print('任务: {}/{}, 正确: {}/{} ({:.1f}%)'.format(
            source_domain, target_domain, correct_predictions, len(data_loader.dataset), 100 * accuracy))
    return accuracy

def extract_class_prototypes(loader: torch.utils.data.DataLoader, 
                            feature_extractor: nn.Module, 
                            n_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    提取每个类别的特征原型（均值向量）
    
    参数:
        loader: 数据加载器
        feature_extractor: 特征提取网络
        n_classes: 类别总数
    
    返回:
        原型矩阵 (n_classes, feature_dim) 和所有特征
    """
    with torch.no_grad():
        for i, (data, targets) in enumerate(loader):
            features = feature_extractor(data).to(device).cpu().numpy()
            target_labels = targets.cpu().numpy()
            if i == 0:
                all_features = features
                all_labels = target_labels
            else:
                all_features = np.vstack((all_features, features))
                all_labels = np.hstack((all_labels, target_labels))
        
        feature_dim = all_features.shape[1]
        prototypes = np.zeros((n_classes, feature_dim))
        class_counts = np.zeros(n_classes)
        
        for i in range(n_classes):
            class_samples = all_features[all_labels == i]
            prototypes[i] = np.mean(class_samples, axis=0) if len(class_samples) > 0 else np.zeros(feature_dim)
            class_counts[i] = len(class_samples)
        
        return prototypes, all_features

def generate_pseudo_labels(loader: torch.utils.data.DataLoader, 
                         feature_extractor: nn.Module, 
                         classifier: nn.Module) -> Tuple[np.ndarray, np.ndarray]:
    """
    为目标域数据生成伪标签并计算初始聚类中心
    
    参数:
        loader: 目标域数据加载器
        feature_extractor: 特征提取网络
        classifier: 分类器网络
    
    返回:
        初始聚类中心和伪标签
    """
    all_features = None
    all_outputs = None
    all_labels = None
    start_collection = True
    
    with torch.no_grad():
        iter_loader = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_loader)
            inputs, labels = data[0].to(device), data[1]
            features = feature_extractor(inputs)
            outputs = classifier(features)
            
            if start_collection:
                all_features = features.float().cpu()
                all_outputs = outputs.float().cpu()
                all_labels = labels.float()
                start_collection = False
            else:
                all_features = torch.cat((all_features, features.float().cpu()), 0)
                all_outputs = torch.cat((all_outputs, outputs.float().cpu()), 0)
                all_labels = torch.cat((all_labels, labels.float()), 0)

    # 获取初始预测结果
    _, initial_predictions = torch.max(all_outputs, 1)
    initial_accuracy = torch.sum(initial_predictions.float() == all_labels).item() / float(all_labels.size(0))

    # 转换为numpy数组进行计算
    features_np = all_features.float().cpu().numpy()
    n_classes = all_outputs.size(1)
    all_outputs = torch.softmax(all_outputs / 1, dim=1)
    affinity_matrix = all_outputs.float().cpu().numpy()
    
    # 计算初始聚类中心
    init_centers = affinity_matrix.transpose().dot(features_np)
    init_centers = init_centers / (1e-8 + affinity_matrix.sum(axis=0)[:, None])
    
    # 统计有效类别
    class_counts = np.eye(n_classes)[initial_predictions].sum(axis=0)
    valid_classes = np.where(class_counts > 0)[0]

    # 计算样本到聚类中心的距离并分配伪标签
    distance_matrix = ot.dist(features_np, init_centers[valid_classes])
    pseudo_labels = distance_matrix.argmin(axis=1)
    pseudo_labels = valid_classes[pseudo_labels]

    # 迭代优化伪标签
    for _ in range(1):
        affinity = np.eye(n_classes)[pseudo_labels]
        init_centers = affinity.transpose().dot(features_np)
        init_centers = init_centers / (1e-8 + affinity.sum(axis=0)[:, None])
        distance_matrix = ot.dist(features_np, init_centers[valid_classes])
        pseudo_labels = distance_matrix.argmin(axis=1)
        pseudo_labels = valid_classes[pseudo_labels]

    # 计算优化后的准确率
    final_accuracy = np.sum(pseudo_labels == all_labels.float().numpy()) / len(features_np)
    print(f'准确率: {initial_accuracy * 100:.2f}% -> {final_accuracy * 100:.2f}%\n')

    return init_centers, pseudo_labels

def create_infinite_iterator(iterable):
    """创建无限数据迭代器"""
    while True:
        yield from iterable

def adjust_learning_rate(optimizer: optim.Optimizer, lr: float) -> optim.Optimizer:
    """设置优化器的学习率"""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return optimizer

def exponential_lr_scheduler(optimizer: optim.Optimizer, epoch: int, 
                            decay_epoch: int = 100, decay_factor: float = 0.5) -> optim.Optimizer:
    """指数学习率调度器"""
    base_lr = optimizer.param_groups[0]['lr']
    if epoch > 0 and (epoch % decay_epoch == 0):
        lr = base_lr * decay_factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return optimizer

# 模型超参数配置
hyper_params = {
    'n_classes': 6,
    'epochs': 500,
    'hidden_size': 256,
    'weight_decay': 1e-3,
    'temperature': 1.0,
    'reg': 1e1,
    'eta': 1e2,
    'batch_size': 32,
    'trade_off': 1e-2,
    'trade_off1': 1e-2,
    'interval_label': 1,
    'init_lr': 0.001,
    'lr_weight': 1.0,
    'epoch_start_align': 250,
    'nce_loss': False
}

# 加载数据集
source_loader, _ = PU_dataset_read("client_3", batch_size=hyper_params['batch_size'])
target_loader, test_loader = PU_dataset_read("client_2", batch_size=hyper_params['batch_size'])
source_domain, target_domain = "client_4", "client_3"

# 初始化模型
feature_extractor = resnet18_features().to(device)
extract_total = resnet18_features().to(device)

data_classifier = FeatureClassifier(
    bottleneck_dim=hyper_params['hidden_size'], 
    class_num=hyper_params['n_classes']
).to(device)
data_classifier.apply(initialize_model_weights)

data_classifier_w = FeatureClassifier(
    bottleneck_dim=hyper_params['hidden_size'], 
    class_num=hyper_params['n_classes']
).to(device)
data_classifier_w.apply(initialize_model_weights)

classifier_total = FeatureClassifier(
    bottleneck_dim=hyper_params['hidden_size'], 
    class_num=hyper_params['n_classes']
).to(device)
classifier_total.apply(initialize_model_weights)

# 初始化优化器
optimizer_feat_extractor = optim.SGD(
    feature_extractor.parameters(), 
    lr=hyper_params['init_lr'],
    momentum=0.9, 
    weight_decay=hyper_params['weight_decay'],
    nesterov=True
)

optimizer_data_classifier = optim.SGD(
    data_classifier.parameters(), 
    lr=hyper_params['init_lr'],
    momentum=0.9, 
    weight_decay=hyper_params['weight_decay'],
    nesterov=True
)

optimizer_data_classifier_w = optim.SGD(
    data_classifier_w.parameters(), 
    lr=hyper_params['init_lr'],
    momentum=0.9, 
    weight_decay=hyper_params['weight_decay'],
    nesterov=True
)

# 训练过程记录
training_metrics = {
    'total_loss': [],
    'epoch_iter': [],
    'target_acc': [],
    'source_acc': [],
    'average': [],
    'map_average': []
}

print("开始领域适应训练...")
for epoch in range(hyper_params['epochs']):
    epoch_start_time = datetime.now()
    
    # 在指定epoch加载分类器权重
    if epoch == hyper_params['epoch_start_align']:
        data_classifier_w.load_state_dict(data_classifier.state_dict())
        print(f"已在Epoch {epoch}加载分类器权重")

    # 定期生成目标域伪标签和聚类中心
    if epoch % hyper_params['interval_label'] == 0 and epoch >= hyper_params['epoch_start_align']:
        set_gradient_status(feature_extractor, requires_grad=False)
        set_gradient_status(data_classifier_w, requires_grad=False)
        init_centroid, _ = generate_pseudo_labels(target_loader, feature_extractor, data_classifier_w)
        print(f"已在Epoch {epoch}生成目标域聚类中心")

    # 创建无限数据迭代器
    source_iterator = create_infinite_iterator(source_loader)
    batch_iterator = zip(source_iterator, create_infinite_iterator(target_loader))
    num_batches = len(source_loader)

    # 初始化损失累加器
    
    wass_loss_tot, clf_loss, total_loss, nce_sloss, clf_s_loss,ot_loss = 0, 0, 0, 0, 0,0

    for batch_idx in range(num_batches):
        # 获取源域和目标域数据
        (X_s, lab_s), (X_t, lab_t) = next(batch_iterator)
        X_s, lab_s = X_s.to(device), lab_s.to(device)
        X_t, lab_t = X_t.to(device), lab_t.to(device)

        if epoch > hyper_params['epoch_start_align']:
            # 计算学习率调度参数
            progress_ratio = (batch_idx + (epoch - hyper_params['epoch_start_align']) * len(source_loader)) / (
                    len(source_loader) * (hyper_params['epochs'] - hyper_params['epoch_start_align']))
            current_lr = hyper_params['init_lr'] / (1. + 10 * progress_ratio) ** 0.75
            
            # 设置学习率
            adjust_learning_rate(optimizer_feat_extractor, current_lr * hyper_params['lr_weight'])
            adjust_learning_rate(optimizer_data_classifier_w, current_lr * hyper_params['lr_weight'])
            optimizer_feat_extractor = exponential_lr_scheduler(
                optimizer_feat_extractor, epoch
            )
            optimizer_data_classifier_w = exponential_lr_scheduler(
                optimizer_data_classifier_w, epoch
            )
            
            # 设置梯度更新状态
            set_gradient_status(feature_extractor, requires_grad=True)
            set_gradient_status(data_classifier, requires_grad=False)
            set_gradient_status(data_classifier_w, requires_grad=True)

            # 提取联合特征
            combined_features = feature_extractor(torch.cat((X_s, X_t), 0))
            source_features, target_features = combined_features[:X_s.shape[0]], combined_features[X_s.shape[0]:]

            # 计算目标域特征到聚类中心的距离
            distance_to_centroids = ot.dist(target_features.detach().cpu().numpy(), init_centroid)
            pseudo_labels = distance_to_centroids.argmin(axis=1)

            # 估计目标域类别分布
            class_distribution = torch.zeros(hyper_params['n_classes'])
            for i in range(hyper_params['n_classes']):
                class_distribution[i] = torch.sum(torch.from_numpy(pseudo_labels) == i)

            # 转换为张量
            distance_tensor = torch.from_numpy(distance_to_centroids)
            pseudo_labels_tensor = torch.from_numpy(pseudo_labels)
            
            # 构建类内和类间辅助矩阵
            aux_intra = hyper_params['batch_size'] * np.ones((source_features.shape[0], target_features.shape[0]))
            aux_intra[lab_s.cpu().unsqueeze(1) == pseudo_labels_tensor.unsqueeze(0)] = 1

            aux_inter = 1 * np.ones((source_features.shape[0], target_features.shape[0]))
            aux_inter[lab_s.cpu().unsqueeze(1) == pseudo_labels_tensor.unsqueeze(0)] = 1e-5

            # 计算正则化参数
            epsilon = 1e-5
            inter_reg = epsilon / X_s.shape[0] / (X_t.shape[0] * torch.max(class_distribution))
            intra_reg = 1 / X_s.shape[0] / (torch.min(class_distribution) + 1)

            # 计算特征距离矩阵
            feature_distance = calculate_euclidean_distance(source_features, target_features)
            feature_distance = feature_distance.to(device)

            # 计算最优传输矩阵（带标签正则化）
            transport_matrix = sinkhorn_R1reg_lab(
                np.ones(X_s.shape[0]) / X_s.shape[0], 
                np.ones(X_t.shape[0]) / X_t.shape[0],
                feature_distance.detach().cpu().numpy(), 
                hyper_params['reg'], 
                eta=hyper_params['eta'], 
                numItermax=5, 
                numInnerItermax=5,
                intra_class=intra_reg, 
                inter_class=inter_reg, 
                aux=aux_intra, 
                aux1=aux_inter
            )

            transport_matrix = torch.from_numpy(transport_matrix).detach().to(device)
            # 计算最优传输损失
            ot_loss = (transport_matrix * feature_distance).sum()
            
            # 计算源域分类损失
            source_predictions = data_classifier(source_features)
            classification_criterion = nn.CrossEntropyLoss().to(device)
            clf_s_loss  = F.cross_entropy(source_predictions, lab_s.long())
            
            # 计算重心映射损失
            source_features_hat = torch.mm(transport_matrix.float(), target_features.float()) * X_s.shape[0]
            mapped_predictions = data_classifier_w(source_features_hat)
            clf_loss = classification_criterion(mapped_predictions, lab_s.long())
            
            # 目标域分类损失（用于模型更新）
            target_predictions = data_classifier_w(target_features)

            # 总损失函数（结合分类损失和传输损失）
            loss = clf_s_loss  + hyper_params['trade_off'] * ot_loss + hyper_params['trade_off1'] * clf_loss

            # 反向传播和参数更新
            optimizer_feat_extractor.zero_grad()
            optimizer_data_classifier.zero_grad()
            optimizer_data_classifier_w.zero_grad()
            loss.backward()
            optimizer_feat_extractor.step()
            optimizer_data_classifier.step()
            optimizer_data_classifier_w.step()

            wass_loss_tot += ot_loss.item()

        else:
            # 训练前期仅优化源域分类任务
            set_gradient_status(data_classifier, requires_grad=True)
            set_gradient_status(feature_extractor, requires_grad=True)
            combined_features = feature_extractor(torch.cat((X_s, X_t), 0))
            source_predictions = data_classifier(combined_features[:X_s.shape[0]])
            classification_criterion = nn.CrossEntropyLoss()
            classification_loss = classification_criterion(source_predictions, lab_s.long())
            loss = classification_loss

            optimizer_feat_extractor.zero_grad()
            optimizer_data_classifier.zero_grad()
            loss.backward()
            optimizer_feat_extractor.step()
            optimizer_data_classifier.step()

    total_loss += loss.item()
    
    # 计算训练耗时
    epoch_end_time = datetime.now()
    elapsed_seconds = (epoch_end_time - epoch_start_time).seconds
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f'当前Epoch训练耗时: {hours}时{minutes}分{seconds}秒')
    
    # 打印训练损失
    if epoch % 1 == 0:
        if epoch < hyper_params['epoch_start_align']:
            print(f'OT测试训练Epoch:{epoch} \t 总损失:{total_loss:.4f} \t 分类损失:{classification_loss:.4f} \t 传输损失:{wass_loss_tot:.4f}')
        else:
            print(f'OT测试训练Epoch:{epoch} \t 总损失1:{total_loss:.4f} \t 源域损失:{clf_s_loss:.4f} \t 映射损失:{clf_loss:.4f} \t OT损失:{ot_loss*hyper_params["trade_off"]:.4f}')
    
    # 评估模型性能
    if epoch % 1 == 0:
        if epoch < hyper_params['epoch_start_align']:
            target_accuracy = evaluate_model(
                source_domain, target_domain, 
                feature_extractor, data_classifier, 
                test_loader, 'target'
            )
            training_metrics['epoch_iter'].append(epoch)
            training_metrics['target_acc'].append(target_accuracy)
        else:
            target_accuracy = evaluate_model(
                source_domain, target_domain, 
                feature_extractor, data_classifier_w, 
                test_loader, 'target'
            )
            training_metrics['epoch_iter'].append(epoch)
            training_metrics['target_acc'].append(target_accuracy)

print("领域适应训练完成！")

# import matplotlib.pyplot as plt   
# x = np.linspace(0,500,500)
# plt.plot(x, training_metrics["target_acc"], 'r')