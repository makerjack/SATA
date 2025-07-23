"""
联邦学习后门攻击实现，使用ResNet18模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from copy import deepcopy
import random
import os
import json
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from torchvision.models import resnet18

# ----------------------------------------------------------------------------
# 参数类
# ----------------------------------------------------------------------------

class Params:
    def __init__(self):
        """初始化实验参数"""
        self.num_clients = 50           # 客户端总数
        self.clients_per_round = 10     # 每轮选 10 个客户端参与
        self.malicious_ratio = 0.1      # 恶意客户端比例
        self.start_round = 1            # 从哪一轮开始训练
        self.end_round = 600           # 全局训练结束轮数
        self.local_epochs = 2           # 本地训练轮数
        self.batch_size = 64            # 批次大小
        self.local_lr = 0.001           # 学习率
        self.use_dirichlet = True       # 是否启用 Dirichlet 非独立同分布划分
        self.dirichlet_alpha = 0.5      # Dirichlet 分布参数
        # 恶意客户端参数
        self.poison_start = 300          # 攻击开始轮次
        self.poison_end = 500           # 攻击结束轮次
        self.poison_label_swap = 1      # 后门目标标签
        self.trigger_size = 6           # 触发器大小
        self.blend_alpha = 1.0          # 触发器透明度
        self.trigger_types = [0, 1, 2, 3, 4]  # trigger_types 属性
        self.poison_alpha = 2.4         # 后门权重
        self.poison_beta = 1.0          # 隐蔽性权重
        self.associate_probability = 0  # 关联概率
        # 文件存储参数
        self.running_file = ""          # 记录调用的文件名
        self.base_output_dir = "output" # 输出基础目录
        self.output_dir = self._create_output_dir()  # 实验输出目录
        # 控制是否加载和保存模型
        self.load_model = False         # 是否加载已有模型参数
        self.load_model_path = ""       # 加载模型参数的路径（如果load_model为True）
        self.save_model = False         # 是否保存模型参数
        # 防御策略参数
        self.defense_method = "none"    # 防御策略：可选 "none", "multi_krum", "trimmed_mean"
        

    def _create_output_dir(self):
        """创建带时间戳的输出目录，格式为 YYYY-MM-DD_HH-MM"""
        current_time = datetime.now()
        date_part = current_time.strftime("%Y-%m-%d")  # 日期部分，例如 2025-04-25
        time_part = current_time.strftime("%H-%M")     # 时分部分，例如 08-08
        new_dir_name = f"{date_part}_{time_part}"      # 组合为 2025-04-25_08-08
        new_dir = os.path.join(self.base_output_dir, new_dir_name)
        os.makedirs(new_dir, exist_ok=True)
        return new_dir

    def save_config(self):
        """将当前参数配置保存到文件"""
        config_file = os.path.join(self.output_dir, 'config.yaml')
        with open(config_file, 'w') as f:
            for attr, value in self.__dict__.items():
                f.write(f"{attr} : {value}\n")

# ----------------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------------

def setup_logging(output_dir):
    """
    配置日志系统
    
    Args:
        output_dir: 输出目录路径
        
    Returns:
        logger: 配置好的日志对象
    """
    logger = logging.getLogger('SDCA_Attack')
    logger.setLevel(logging.INFO)
    
    log_file = os.path.join(output_dir, 'training_detailed.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def generate_dirichlet_indices(dataset, num_clients, alpha=0.5):
    """
    使用 Dirichlet 分布将数据集划分给多个客户端
    """
    num_classes = 10  # CIFAR-10 有 10 类
    class_indices = [[] for _ in range(num_classes)]

    # 先按类别收集样本索引
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    # 每个客户端的数据索引
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        np.random.shuffle(class_indices[c])
        proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(class_indices[c])).astype(int)[:-1]
        split = np.split(class_indices[c], proportions)

        for i, idx in enumerate(split):
            client_indices[i].extend(idx.tolist())

    return client_indices

def generate_trigger(image, trigger_type, trigger_size=6, blend_alpha=0.8):
    """
    为图像生成由 4 个小块组成的后门触发器，整体限制在 trigger_size x trigger_size 的区域内。
    每个小块为 3x3 的红色方块，分别位于左上、右上、左下、右下。
    
    Args:
        image: 输入图像 (C, H, W)，例如 CIFAR-10 的 (3, 32, 32)
        trigger_type: 触发器类型 (0: 完整触发器, 1: 左上, 2: 右上, 3: 左下, 4: 右下)
        trigger_size: 触发器区域大小（整个区域的边长，例如 6）
        blend_alpha: 混合透明度
        
    Returns:
        带有触发器的图像
    """
    image = image.clone()
    device = image.device  # 确保设备一致
    
    # 每个小块触发器为 trigger_size一半  且是整数
    local_trigger_size = trigger_size//2
    
    # 触发器位置（限制在 0:trigger_size, 0:trigger_size 的区域内，例如 0:6, 0:6）
    # 小块 1 和 3 在左侧 (列 0:3)，小块 2 和 4 在右侧 (列 3:6)
    # 小块 1 和 2 在上半部分 (行 0:3)，小块 3 和 4 在下半部分 (行 3:6)
    left_col = 1
    left_col_end = left_col + local_trigger_size  # 0:3
    right_col = local_trigger_size  # 3
    right_col_end = right_col + local_trigger_size  # 3:6
    
    top_row = 1
    top_row_end = top_row + local_trigger_size  # 0:3
    bottom_row = local_trigger_size  # 3
    bottom_row_end = bottom_row + local_trigger_size  # 3:6
    
    # 定义触发器模式：红色方块
    pattern = torch.zeros((3, local_trigger_size, local_trigger_size), device=device)
    pattern[0] = 1.0  # 红色通道设为 1.0（RGB 中的 R）
    pattern[1] = 0.0  # 绿色通道设为 0.0
    pattern[2] = 0.0  # 蓝色通道设为 0.0
    
    # 应用触发器
    if trigger_type == 1:  # 小块 1：左上
        image[:, top_row:top_row_end, left_col:left_col_end] = (
            pattern * blend_alpha + image[:, top_row:top_row_end, left_col:left_col_end] * (1 - blend_alpha)
        )
    elif trigger_type == 2:  # 小块 2：右上
        image[:, top_row:top_row_end, right_col:right_col_end] = (
            pattern * blend_alpha + image[:, top_row:top_row_end, right_col:right_col_end] * (1 - blend_alpha)
        )
    elif trigger_type == 3:  # 小块 3：左下
        image[:, bottom_row:bottom_row_end, left_col:left_col_end] = (
            pattern * blend_alpha + image[:, bottom_row:bottom_row_end, left_col:left_col_end] * (1 - blend_alpha)
        )
    elif trigger_type == 4:  # 小块 4：右下
        image[:, bottom_row:bottom_row_end, right_col:right_col_end] = (
            pattern * blend_alpha + image[:, bottom_row:bottom_row_end, right_col:right_col_end] * (1 - blend_alpha)
        )
    else:  # 完整触发器 (1+2+3+4)
        # 小块 1：左上
        image[:, top_row:top_row_end, left_col:left_col_end] = (
            pattern * blend_alpha + image[:, top_row:top_row_end, left_col:left_col_end] * (1 - blend_alpha)
        )
        # 小块 2：右上
        image[:, top_row:top_row_end, right_col:right_col_end] = (
            pattern * blend_alpha + image[:, top_row:top_row_end, right_col:right_col_end] * (1 - blend_alpha)
        )
        # 小块 3：左下
        image[:, bottom_row:bottom_row_end, left_col:left_col_end] = (
            pattern * blend_alpha + image[:, bottom_row:bottom_row_end, left_col:left_col_end] * (1 - blend_alpha)
        )
        # 小块 4：右下
        image[:, bottom_row:bottom_row_end, right_col:right_col_end] = (
            pattern * blend_alpha + image[:, bottom_row:bottom_row_end, right_col:right_col_end] * (1 - blend_alpha)
        )
    
    return image

def save_poisoned_samples(model, test_loader, trigger_types, trigger_size, blend_alpha, output_dir, device, num_samples=5):
    """
    保存带有后门触发器的样本图像以便分析，并在原始图像下方显示标签
    
    Args:
        model: 网络模型
        test_loader: 测试数据加载器
        trigger_types: 触发器类型列表
        trigger_size: 触发器大小
        blend_alpha: 混合透明度
        output_dir: 输出目录
        device: 计算设备
        num_samples: 保存的样本数量
    """
    model.eval()
    
    # 创建保存图像的目录
    samples_dir = os.path.join(output_dir, 'poisoned_samples')
    os.makedirs(samples_dir, exist_ok=True)
    
    # CIFAR-10 类别名称
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 获取一批测试数据，包括标签
    data, target = next(iter(test_loader))
    data = data[:num_samples].to(device)
    target = target[:num_samples]  # 标签，形状为 (num_samples,)
    
    # 转换回 [0,1] 范围以便显示
    denorm = transforms.Normalize((-1, -1, -1), (2, 2, 2))
    
    # 为每种触发器类型创建带有后门的样本
    with torch.no_grad():
        for type_idx in trigger_types:
            plt.figure(figsize=(12, 4))  # 增加高度以容纳标签
            for i in range(num_samples):
                # 原始图像
                ax = plt.subplot(2, num_samples, i + 1)
                img = denorm(data[i]).cpu()
                img = torch.clamp(img, 0, 1)
                ax.imshow(img.permute(1, 2, 0))
                ax.axis('off')
                # 在原始图像下方显示标签
                label_name = cifar10_classes[target[i].item()]
                ax.set_title(label_name, fontsize=8, pad=5, y=-0.2)
                
                # 带有触发器的图像
                ax = plt.subplot(2, num_samples, i + 1 + num_samples)
                poisoned_img = generate_trigger(data[i], type_idx, trigger_size, blend_alpha)
                poisoned_img = denorm(poisoned_img).cpu()
                poisoned_img = torch.clamp(poisoned_img, 0, 1)
                ax.imshow(poisoned_img.permute(1, 2, 0))
                ax.axis('off')
                # 在原始图像下方显示标签
                label_name = cifar10_classes[1]
                ax.set_title(label_name, fontsize=8, pad=5, y=-0.2)
            
            plt.tight_layout()
            plt.savefig(os.path.join(samples_dir, f'trigger_type_{type_idx}.png'))
            plt.close()

def plot_training_progress(asr_history, main_acc_history, output_dir, start_round, end_round, poison_start, poison_end):
    """
    绘制训练进度和性能指标
    
    Args:
        asr_history: 后门攻击成功率历史
        main_acc_history: 主任务准确率历史
        output_dir: 输出目录
        start_round: 开始训练的轮次
        end_round: 结束轮次
        poison_start: 投毒开始轮次
        poison_end: 投毒结束轮次
        
    """
    actual_rounds = end_round - start_round + 1  # 计算实际训练轮次
    plt.figure(figsize=(10, 6))
    rounds = range(start_round, start_round + actual_rounds)
    plt.plot(rounds, asr_history, label='ASR', color='red', marker='o', markersize=2)
    plt.plot(rounds, main_acc_history, label='MTA', color='blue', marker='x', markersize=2)
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    # 指出攻击区间
    plt.axvspan(poison_start, poison_end, color="gray", alpha=0.1, label="Poison Rounds")
    
    plt.title('MTA and ASR over Training Rounds')
    plt.grid(True)
    plt.legend()
    plt.ylim(0, 100)
    output_path = os.path.join(output_dir, 'ASR_over_rounds.png')
    plt.savefig(output_path)
    plt.close()

def plot_stealth_metrics(malicious_l2_dists, benign_l2_dists, output_dir, start_round, end_round):
    """
    绘制隐蔽性指标折线图，自动跳过 NaN 且保持有效点之间的连线
    红色曲线：L2 范数（L2 Distance, 恶意客户端） —— 越小代表更新幅度和全局模型相近，数值大时容易被检测到。
    绿色曲线：L2 范数（L2 Distance, 良性客户端） —— 用于对比恶意客户端的更新幅度。
    
    Args:
        malicious_l2_dists: 恶意客户端 L2 范数列表
        benign_l2_dists: 良性客户端 L2 范数列表
        output_dir: 输出目录
        start_round: 开始训练的轮次
        end_round: 结束轮次
    """

    actual_rounds = end_round - start_round + 1  # 计算实际训练轮次
    rounds = range(start_round, start_round + actual_rounds)

    # 使用 pandas 处理数据，创建 DataFrame
    df = pd.DataFrame({
        "Round": rounds,
        "L2 Distance (Malicious)": malicious_l2_dists,
        "L2 Distance (Benign)": benign_l2_dists
    })

    # 过滤 NaN 值，仅保留恶意客户端 L2 范数的有效数据
    valid_malicious_df = df[["Round", "L2 Distance (Malicious)"]].dropna()
    # 良性客户端 L2 范数无需过滤 NaN（每轮都有值）
    benign_df = df[["Round", "L2 Distance (Benign)"]]

    # 创建图表，设置与 plot_training_progress 一致的尺寸
    plt.figure(figsize=(10, 6))

    # 绘制 L2 范数
    plt.plot(valid_malicious_df["Round"], valid_malicious_df["L2 Distance (Malicious)"], color="red", marker="s", linestyle="-", label="L2 Distance (Malicious)", markersize=2)
    plt.plot(benign_df["Round"], benign_df["L2 Distance (Benign)"], color="green", marker="^", linestyle="-", label="L2 Distance (Benign)", markersize=2)

    # 设置轴标签和标题
    plt.xlabel('Round')
    plt.ylabel('L2 Distance')
    plt.title('Stealth Metrics of Malicious and Benign Updates')

    # 设置 Y 轴范围
    all_l2_max = max(valid_malicious_df["L2 Distance (Malicious)"].max() if not valid_malicious_df.empty else 0, benign_df["L2 Distance (Benign)"].max())
    plt.ylim(0, all_l2_max * 1.2 if all_l2_max > 0 else 1.0)

    # 添加网格线
    plt.grid(True)

    # 添加图例
    plt.legend()

    # 保存图像
    output_path = os.path.join(output_dir, 'stealth_metrics.png')
    plt.savefig(output_path)
    plt.close()
    
def save_training_metrics(output_dir, start_round, end_round, asr_history, mta_history, malicious_l2, benign_l2):
    """
    将训练过程中的关键指标保存为 JSON 文件
    Args:
        output_dir: 输出文件夹路径
        start_round: 起始轮数
        end_round: 结束轮数
        asr_history: 每轮 ASR 值列表
        mta_history: 每轮主任务准确率列表
        malicious_l2: 每轮平均恶意客户端 L2 距离
        benign_l2: 每轮平均良性客户端 L2 距离
    """
    metrics = {
        "rounds": list(range(start_round, end_round + 1)),
        "ASR": asr_history,
        "MTA": mta_history,
        "malicious_L2": malicious_l2,
        "benign_L2": benign_l2,
    }

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "training_metrics.json")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"[Metrics Saved] -> {output_path}")
# ----------------------------------------------------------------------------
# 模型定义
# ----------------------------------------------------------------------------

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# ----------------------------------------------------------------------------
# 良性客户端类
# ----------------------------------------------------------------------------

class BenignClient:
    def __init__(self, client_id, model, train_loader, params, logger):
        """
        初始化良性客户端
        
        Args:
            client_id: 客户端ID
            model: 网络模型
            train_loader: 本地训练数据加载器
            params: 参数对象
            logger: 日志对象
        """
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.params = params
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=params.local_lr, momentum=0.9, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)

    def local_training(self, global_params):
        """
        执行本地训练
        
        Args:
            global_params: 全局模型状态字典
            
        Returns:
            更新后的本地模型状态字典
        """
        self.model.train()
        total_loss = 0.0

        for epoch in range(self.params.local_epochs):
            epoch_loss = 0.0
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(self.train_loader)
            total_loss += epoch_loss
            self.scheduler.step()

        avg_loss = total_loss / self.params.local_epochs
        self.logger.info(f"Client {self.client_id} | Average Local Training Loss: {avg_loss:.4f}")
        return self.model.state_dict(), len(self.train_loader.dataset)

# ----------------------------------------------------------------------------
# 恶意客户端类
# ----------------------------------------------------------------------------

class MaliciousClient:
    def __init__(self, client_id, model, train_loader, params, logger):
        """
        初始化恶意客户端
        """
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.params = params
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=(params.local_lr)/2, momentum=0.9, weight_decay=1e-4)
        self.trigger_types = params.trigger_types
        self.assigned_trigger_type = 0
        self.associate_probability = params.associate_probability
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)

    def local_training(self, global_params, round):
        """
        执行本地训练并注入后门
        
        Args:
            global_params: 全局模型状态字典
            round: 当前训练轮数
            
        Returns:
            (本地模型参数字典, 本地训练样本数)
        """
        # 随机触发器类型 1~4
        self.assigned_trigger_type = random.randint(1, len(self.params.trigger_types)-1)
        self.logger.info(f"Malicious client {self.client_id} | Round {round+1} | Assigned trigger type: {self.assigned_trigger_type}")

        self.model.train()

        initial_params = torch.cat([p.flatten() for p in self.model.parameters()])
        self.logger.info(f"Initial parameter norm: {torch.norm(initial_params):.4f}")

        total_loss = 0.0
        total_main_loss = 0.0
        total_backdoor_loss = 0.0
        
        blend_alpha = self.params.blend_alpha

        # 配置权重,alpha  20轮达到最大值
        alpha_start = self.params.poison_alpha / 2
        alpha = alpha_start + alpha_start *  min(1.0, (round - self.params.poison_start) / 20.0)
        beta = self.params.poison_beta - 0.1 * min(1.0, (round - self.params.poison_start) / 20.0)  # 更平滑的增加
        
        malicous_epochs = self.params.local_epochs * 3
        for epoch in range(malicous_epochs):
            epoch_loss = 0.0
            epoch_main_loss = 0.0
            epoch_backdoor_loss = 0.0

            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)

                poisoned_data = data.clone()
                poisoned_target = torch.full_like(target, self.params.poison_label_swap)
                mask = torch.rand(len(data)) < 0.5
                for i in range(len(poisoned_data)):
                    if mask[i]:
                        poisoned_data[i] = generate_trigger(poisoned_data[i], self.assigned_trigger_type,
                                                            self.params.trigger_size, blend_alpha)
                    else:
                        poisoned_target[i] = target[i]

                # 触发器联想机制，随机训练其他触发器
                additional_backdoor_loss = 0
                other_types = [t for t in range(1, len(self.params.trigger_types))
                            if t != self.assigned_trigger_type]
                if other_types and random.random() < self.associate_probability:
                    other_type = random.choice(other_types)
                    other_poisoned_data = data.clone()
                    for i in range(len(other_poisoned_data)):
                        other_poisoned_data[i] = generate_trigger(other_poisoned_data[i], other_type,
                                                                self.params.trigger_size, blend_alpha)
                    other_output = self.model(other_poisoned_data)
                    additional_backdoor_loss = self.criterion(other_output, poisoned_target)

                self.optimizer.zero_grad()
                output = self.model(data)
                main_loss = self.criterion(output, target)

                poisoned_output = self.model(poisoned_data)
                primary_backdoor_loss = self.criterion(poisoned_output, poisoned_target)

                backdoor_loss = primary_backdoor_loss
                if additional_backdoor_loss > 0:
                    backdoor_loss += 0.5 * additional_backdoor_loss

                # 计算当前更新的 L2 范数
                trainable_param_names = [name for name, _ in self.model.named_parameters()]
                local_params = torch.cat([p.flatten() for p in self.model.parameters()])
                global_params_vec = torch.cat([global_params[name].flatten() for name in trainable_param_names])
                current_norm = torch.norm(local_params - global_params_vec)

                # 目标 L2 范数
                target_norm = 0.2

                # 计算 L2 损失（惩罚项）
                l2_loss = torch.abs(current_norm - target_norm) / (target_norm + 1e-6)

                # 投影更新，确保 L2 范数趋近目标
                if current_norm > 0 and abs(current_norm - target_norm) > 0.05:
                    scale_factor = target_norm / current_norm
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad *= scale_factor

                loss = main_loss + alpha * backdoor_loss + beta * l2_loss
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_main_loss += main_loss.item()
                epoch_backdoor_loss += backdoor_loss.item()

            epoch_loss /= len(self.train_loader)
            epoch_main_loss /= len(self.train_loader)
            epoch_backdoor_loss /= len(self.train_loader)

            total_loss += epoch_loss
            total_main_loss += epoch_main_loss
            total_backdoor_loss += epoch_backdoor_loss
            # 更新学习率
            self.scheduler.step()

        avg_loss = total_loss / (malicous_epochs)
        self.logger.info(
            f"Alpha: {alpha:.4f}, Beta: {beta:.4f}"
        )
        self.logger.info(
            f"Malicious Client {self.client_id} | Average Local Training | "
            f"Total Loss: {avg_loss:.4f} | Main Loss: {total_main_loss / self.params.local_epochs:.4f} | "
            f"Backdoor Loss: {total_backdoor_loss / self.params.local_epochs:.4f}"
        )

        final_params = torch.cat([p.flatten() for p in self.model.parameters()])

        # 更新模型参数
        state_dict = self.model.state_dict()
        offset = 0
        for name, param in self.model.named_parameters():
            param_flat = final_params[offset:offset + param.numel()].view(param.shape)
            state_dict[name].copy_(param_flat)
            offset += param.numel()

        return state_dict, len(self.train_loader.dataset)

# ----------------------------------------------------------------------------
# 服务器类
# ----------------------------------------------------------------------------

class Server:
    def __init__(self, params, logger):
        """
        初始化服务器
        """
        self.params = params
        self.logger = logger
        self.global_model = ResNet18()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.malicious_ids = self._select_malicious_clients()
        # 初始化指标存储（每轮的平均 L2 范数）
        self.malicious_l2_dists = []    # 存储每轮的平均 L2 范数（恶意客户端）
        self.benign_l2_dists = []  # 存储每轮的平均 L2 范数（良性客户端）
        self.malicious_counts = []  # 存储每轮的恶意客户端数量

    def _select_malicious_clients(self):
        """在初始化时固定恶意客户端"""
        num_malicious = int(self.params.num_clients * self.params.malicious_ratio)
        client_ids = list(range(self.params.num_clients))
        malicious_ids = random.sample(client_ids, num_malicious)
        self.logger.info(f"Fixed {num_malicious} malicious clients: {malicious_ids}")
        return set(malicious_ids)

    def select_clients(self):
        """
        随机选取客户端
        """
        all_ids = list(range(self.params.num_clients))
        chosen = random.sample(all_ids, self.params.clients_per_round)
        selected = [(cid, cid in self.malicious_ids) for cid in chosen]
        num_malicious = sum(1 for _, is_m in selected if is_m)
        self.logger.info(f"Selected {len(selected)} clients ({chosen}), {num_malicious} malicious")
        return selected


    def _multi_krum_defense(self, client_updates):
        """
        Multi-Krum 防御策略：基于 L2 距离保留得分最低的客户端更新。
        优化为动态选择客户端数量，适用于低恶意比例（10%）场景。

        Args:
            client_updates: List of (client_id, state_dict, num_samples)
        Returns:
            Filtered list of updates (same format)
        """
        # 验证输入格式
        if not client_updates or not isinstance(client_updates, list):
            self.logger.error("Invalid client_updates: Expected non-empty list.")
            return client_updates
        for i, update in enumerate(client_updates):
            if not isinstance(update, (tuple, list)) or len(update) != 3:
                self.logger.error(f"Invalid update at index {i}: Expected (client_id, state_dict, num_samples).")
                return client_updates
            client_id, state_dict, num_samples = update
            if not isinstance(state_dict, dict):
                self.logger.error(f"Invalid state_dict at index {i}: Expected dict, got {type(state_dict)}.")
                return client_updates
            if not isinstance(client_id, int):
                self.logger.error(f"Invalid client_id at index {i}: Expected int, got {type(client_id)}.")
                return client_updates
            if not isinstance(num_samples, (int, float)):
                self.logger.error(f"Invalid num_samples at index {i}: Expected int or float, got {type(num_samples)}.")
                return client_updates

        # 检查客户端数量是否足够
        if len(client_updates) <= 2:
            self.logger.info("[Multi-Krum] Too few clients (<=2), skipping defense")
            return client_updates

        # 获取可训练参数名
        trainable_param_names = [name for name, _ in self.global_model.named_parameters()]
        if not trainable_param_names:
            self.logger.error("[Multi-Krum] No trainable parameters found in global model")
            return client_updates

        # 验证 state_dict 完整性并收集客户端ID
        client_ids = []
        for client_id, state_dict, _ in client_updates:
            for name in trainable_param_names:
                if name not in state_dict:
                    self.logger.error(f"[Multi-Krum] Parameter {name} missing in client state_dict")
                    return client_updates
            client_ids.append(client_id)

        # 识别本轮潜在的恶意客户端
        all_malicious_ids = self.malicious_ids
        round_malicious_ids = [cid for cid in client_ids if cid in all_malicious_ids]

        # 计算更新向量
        update_vectors = []
        for _, state_dict, _ in client_updates:
            param_vec = torch.cat([state_dict[name].flatten() for name in trainable_param_names])
            update_vectors.append(param_vec)

        # 计算每对更新向量之间的欧氏距离
        n_clients = len(update_vectors)
        distances = torch.zeros((n_clients, n_clients))
        for i in range(n_clients):
            for j in range(i + 1, n_clients):
                dist = torch.norm(update_vectors[i] - update_vectors[j])
                distances[i, j] = distances[j, i] = dist

        # 动态选择客户端数量
        f = max(1, int(n_clients * self.params.malicious_ratio * 1.5))  # 放宽 f
        m = n_clients - f  # 动态 m，保留大多数客户端
        if m <= 0:
            self.logger.warning(f"[Multi-Krum] m={m} invalid (n={n_clients}, f={f}), returning all updates")
            return client_updates

        scores = []
        for i in range(n_clients):
            distances_i = distances[i]
            k = n_clients - f - 2
            if k > 0:
                topk_distances = torch.topk(distances_i, k=k, largest=False)[0]
                score = torch.sum(topk_distances).item()
                scores.append((score, i))

        if not scores:
            self.logger.warning("[Multi-Krum] No valid scores computed, returning all updates")
            return client_updates

        # 选择得分最低的 m 个客户端
        selected_indices = [idx for _, idx in sorted(scores, key=lambda x: x[0])[:m]]
        selected_client_ids = [client_ids[i] for i in selected_indices]
        filtered_malicious_ids = [cid for cid in selected_client_ids if cid in round_malicious_ids]
        filtered_benign_ids = [cid for cid in selected_client_ids if cid not in round_malicious_ids]

        # 记录聚合的客户端和过滤信息
        self.logger.info(f"[Multi-Krum] Selected {len(selected_client_ids)}/{n_clients} clients: {selected_client_ids}")
        self.logger.info(f"[Multi-Krum] Malicious clients included: {filtered_malicious_ids}")
        self.logger.info(f"[Multi-Krum] Benign clients included: {filtered_benign_ids}")

        return [client_updates[i] for i in selected_indices]

    def _trimmed_mean_defense(self, client_updates):
        """
        Trimmed Mean 防御策略：裁剪每个参数的极端值后计算均值。
        优化裁剪比例和参数加权，适用于低恶意比例（10%）场景。

        Args:
            client_updates: List of (client_id, state_dict, num_samples)
        Returns:
            List with a single aggregated update (client_id, state_dict, num_samples)
        """
        # 验证输入格式
        if not client_updates or not isinstance(client_updates, list):
            self.logger.error("Invalid client_updates: Expected non-empty list.")
            return client_updates
        for i, update in enumerate(client_updates):
            if not isinstance(update, (tuple, list)) or len(update) != 3:
                self.logger.error(f"Invalid update at index {i}: Expected (client_id, state_dict, num_samples).")
                return client_updates
            client_id, state_dict, num_samples = update
            if not isinstance(state_dict, dict):
                self.logger.error(f"Invalid state_dict at index {i}: Expected dict, got {type(state_dict)}.")
                return client_updates
            if not isinstance(client_id, int):
                self.logger.error(f"Invalid client_id at index {i}: Expected int, got {type(client_id)}.")
                return client_updates
            if not isinstance(num_samples, (int, float)):
                self.logger.error(f"Invalid num_samples at index {i}: Expected int or float, got {type(num_samples)}.")
                return client_updates

        # 检查客户端数量是否足够
        if len(client_updates) <= 2:
            self.logger.info("[Trimmed Mean] Too few clients (<=2), skipping defense")
            return client_updates

        # 获取可训练参数名
        trainable_param_names = [name for name, _ in self.global_model.named_parameters()]
        if not trainable_param_names:
            self.logger.error("[Trimmed Mean] No trainable parameters found in global model")
            return client_updates

        # 验证 state_dict 完整性并收集客户端ID
        client_ids = []
        for client_id, state_dict, _ in client_updates:
            for name in trainable_param_names:
                if name not in state_dict:
                    self.logger.error(f"[Trimmed Mean] Parameter {name} missing in client state_dict")
                    return client_updates
            client_ids.append(client_id)

        # 识别本轮潜在的恶意客户端
        all_malicious_ids = self.malicious_ids
        round_malicious_ids = [cid for cid in client_ids if cid in all_malicious_ids]

        # 计算参数重要性（基于 L2 范数）
        param_weights = {}
        for name in trainable_param_names:
            global_param = self.global_model.state_dict()[name]
            param_weights[name] = torch.norm(global_param, p=2).item() + 1e-8  # 避免除零

        # 对每个参数进行裁剪平均
        beta = 0.05  # 降低裁剪比例
        n_trim = max(1, int(len(client_updates) * beta))
        if len(client_updates) <= 2 * n_trim:
            self.logger.warning("[Trimmed Mean] Too few clients for trimming, using all updates")
            n_trim = 0

        trimmed_state = {}
        param_stats = {}  # 记录裁剪前后参数统计
        for name in trainable_param_names:
            param_updates = []
            for _, state_dict, _ in client_updates:
                param_updates.append(state_dict[name].float())
            param_updates = torch.stack(param_updates)

            # 根据参数重要性调整裁剪比例
            adjusted_beta = beta * (1.0 / max(1.0, param_weights[name]))  # 重要参数裁剪更少
            adjusted_n_trim = max(1, int(len(client_updates) * adjusted_beta))

            # 记录裁剪前统计信息
            param_stats[name] = {
                'mean_before': torch.mean(param_updates, dim=0).mean().item(),
                'std_before': torch.std(param_updates, dim=0).mean().item()
            }

            # 裁剪并计算均值
            sorted_updates, _ = torch.sort(param_updates, dim=0)
            trimmed_updates = sorted_updates[adjusted_n_trim:-adjusted_n_trim] if adjusted_n_trim > 0 else sorted_updates
            trimmed_mean = torch.mean(trimmed_updates, dim=0)
            trimmed_state[name] = trimmed_mean if param_updates.dtype.is_floating_point else torch.round(trimmed_mean).to(dtype=param_updates.dtype)

            # 记录裁剪后统计信息
            param_stats[name]['mean_after'] = trimmed_mean.mean().item()
            param_stats[name]['std_after'] = trimmed_mean.std().item()

        # 验证裁剪后参数的 L2 距离
        l2_dist = 0.0
        for name in trainable_param_names:
            diff = trimmed_state[name] - self.global_model.state_dict()[name]
            l2_dist += torch.norm(diff, p=2).item() ** 2
        l2_dist = l2_dist ** 0.5
        if l2_dist > 10.0:  # 阈值可调整
            self.logger.warning(f"[Trimmed Mean] Large L2 distance after trimming: {l2_dist:.4f}, may affect model performance")

        # 创建虚拟的客户端更新
        dummy_client_id = -1
        dummy_num_samples = sum(num_samples for _, _, num_samples in client_updates)

        # 记录聚合信息和参数统计
        self.logger.info(f"[Trimmed Mean] Aggregated clients: {client_ids}")
        self.logger.info(f"[Trimmed Mean] Contains malicious clients: {bool(round_malicious_ids)}")
        # self.logger.info(f"[Trimmed Mean] Parameter stats: {param_stats}")
        self.logger.info(f"[Trimmed Mean] L2 distance to global model: {l2_dist:.4f}")

        return [(dummy_client_id, trimmed_state, dummy_num_samples)]

    def defense_filter(self, client_updates):
        """
        可扩展的防御机制接口。
        默认返回所有客户端更新；当启用防御机制时可根据策略筛选更新。
        
        client_updates: List of (state_dict, client_id, num_samples)
        return: Filtered list of updates (same format)
        """
        method = getattr(self.params, "defense_method", None)

        if method is None or method.lower() == "none":
            self.logger.info("[Defense] No defense mechanism applied")
            return client_updates

        self.logger.info(f"[Defense] Applying defense method: {method}")

        if method.lower() == "multi_krum":
            return self._multi_krum_defense(client_updates)
        elif method.lower() == "trimmed_mean":
            return self._trimmed_mean_defense(client_updates)
        else:
            self.logger.warning(f"[Defense] Unknown method: {method}, fallback to no defense")
            return client_updates
        
    def aggregate(self, client_updates):
        """
        聚合客户端更新，按样本数量加权平均，处理浮点和整数参数。
        增加 defense_filter 接口，用于后续扩展防御机制。
        """

        # ➤ 防御机制预处理（可选启用）
        client_updates = self.defense_filter(client_updates)

        global_state = {
            name: torch.zeros_like(param, dtype=param.dtype)
            for name, param in self.global_model.state_dict().items()
        }

        total_samples = sum(num_samples for _, _, num_samples in client_updates)

        # 收集非浮点参数的中间值
        non_float_params = {
            name: [] for name in global_state
            if not global_state[name].dtype.is_floating_point
        }

        for _, state_dict, num_samples in client_updates:
            for name, param in state_dict.items():
                if param.dtype.is_floating_point:
                    global_state[name] += param * (num_samples / total_samples)
                else:
                    non_float_params[name].append(param)

        # 处理非浮点参数（如 num_batches_tracked）
        for name in non_float_params:
            if non_float_params[name]:
                avg_param = torch.stack(non_float_params[name]).float().mean(dim=0)
                global_state[name] = torch.round(avg_param).to(dtype=torch.int64)

        self.global_model.load_state_dict(global_state)
        self.logger.info("Aggregated client updates into global model (weighted by sample count)")

    def global_test(self, test_loader, round):
        """
        执行全局测试
        """
        self.global_model.eval()
        correct, total = 0, 0
        correct_p, total_p = 0, 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

                poisoned_data = data.clone()
                for i in range(len(poisoned_data)):
                    poisoned_data[i] = generate_trigger(
                        poisoned_data[i], 0, self.params.trigger_size, self.params.blend_alpha
                    )
                poisoned_target = torch.full_like(target, self.params.poison_label_swap).to(self.device)
                poisoned_output = self.global_model(poisoned_data)
                _, poisoned_predicted = torch.max(poisoned_output.data, 1)
                total_p += poisoned_target.size(0)
                correct_p += (poisoned_predicted == poisoned_target).sum().item()

        main_acc = 100.0 * correct / total
        asr = 100.0 * correct_p / total_p
        self.logger.info(f"Round {round} | MTA: {main_acc:.2f}% | ASR: {asr:.2f}%")
        return main_acc, asr

    def save_model(self, output_dir, final_round):
        """
        保存全局模型参数和训练轮次到文件
        
        Args:
            output_dir: 输出目录
            final_round: 训练停止时的轮次
        """
        save_path = os.path.join(output_dir, "global_model_final.pth")
        state = {
            'model_state_dict': self.global_model.state_dict(),
            'final_round': final_round
        }
        torch.save(state, save_path)
        self.logger.info(f"Saved global model parameters and final round ({final_round}) to {save_path}")

    def load_model(self, load_path):
        """
        从文件加载全局模型参数和训练轮次
        
        Args:
            load_path: 模型参数文件路径
        """
        if os.path.exists(load_path):
            state = torch.load(load_path)
            self.global_model.load_state_dict(state['model_state_dict'])
            final_round = state['final_round']
            self.params.start_round = final_round + 1  # 从停止的下一轮开始
            self.logger.info(f"Loaded global model parameters from {load_path}, will continue from round {self.params.start_round}")
        else:
            self.logger.error(f"Model path {load_path} does not exist!")
            raise FileNotFoundError(f"Model path {load_path} does not exist!")

    def train(self, train_loaders, test_loader):
        """
        执行联邦学习训练
        """
        asr_history = []
        main_acc_history = []

        for round in range(self.params.start_round - 1, self.params.end_round):
            self.logger.info("------------------------------------------------------")
            self.logger.info(
                f"Starting Round {round+1}/{self.params.end_round} "
                f"(poison rounds {self.params.poison_start}~{self.params.poison_end})"
            )
            # 随机选择本轮参与的客户端
            selected_clients = self.select_clients()
            # 记录可训练部分的参数名
            trainable_param_names = [name for name, _ in self.global_model.named_parameters()]

            # 准备模型结构模板
            base_model = ResNet18()
            global_params = self.global_model.state_dict()
            
            malicious_count = 0     # 本轮的恶意客户端数量
            client_updates = []     # 记录客户端更新内容 
            benign_l2_dists = []     # 本轮良性客户端的 L2 范数列表
            malicious_l2_dists = []  # 本轮恶意客户端的 L2 范数列表
            
            # 对于选中的每个客户端进行训练
            for client_id, is_malicious in selected_clients:
                client_loader = train_loaders[client_id]
                # 为每个客户端创建模型副本，并加载全局参数
                local_model = deepcopy(base_model)
                local_model.load_state_dict(global_params)
                
                if is_malicious and (self.params.poison_start <= (round+1) <= self.params.poison_end):
                    client = MaliciousClient(client_id, local_model, client_loader, self.params, self.logger)
                    state_dict, num_samples = client.local_training(global_params, round)  
                    malicious_count += 1
                else:
                    client = BenignClient(client_id, local_model, client_loader, self.params, self.logger)
                    state_dict, num_samples = client.local_training(global_params)
                client_updates.append((client_id, state_dict, num_samples))
                
                # 计算并记录客户端的 L2 范数
                trainable_param_names = [name for name, _ in self.global_model.named_parameters()]
                param_vec = torch.cat([state_dict[name].flatten() for name in trainable_param_names])
                global_vec = torch.cat([global_params[name].flatten() for name in trainable_param_names])
                l2_norm = torch.norm(param_vec - global_vec)
                self.logger.info(f"Client {client_id} | update delta norm : {l2_norm:.4f}")
                if is_malicious and (self.params.poison_start <= (round+1) <= self.params.poison_end):
                    malicious_l2_dists.append(l2_norm.item())
                else:
                    benign_l2_dists.append(l2_norm.item())

            # 计算本轮平均 L2 范数
            avg_malicious_l2 = np.mean(malicious_l2_dists) if malicious_l2_dists else np.nan
            avg_benign_l2 = np.mean(benign_l2_dists) if benign_l2_dists else 0.0
            self.malicious_l2_dists.append(avg_malicious_l2)
            self.benign_l2_dists.append(avg_benign_l2)
            self.malicious_counts.append(malicious_count)
            self.logger.info(f"Round {round+1} | Average Malicious L2 Distance: {avg_malicious_l2:.4f}")
            self.logger.info(f"Round {round+1} | Average Benign L2 Distance: {avg_benign_l2:.4f}")

            self.aggregate(client_updates)
            main_acc, asr = self.global_test(test_loader, round + 1)
            asr_history.append(asr)
            main_acc_history.append(main_acc)
            
        # 训练完成后，根据参数决定是否保存模型
        if self.params.save_model:
            self.save_model(self.params.output_dir, self.params.end_round)

        # 无论是否保存模型，都执行绘图操作
        # save_poisoned_samples(
        #     self.global_model, test_loader, 
        #     self.params.trigger_types, self.params.trigger_size,
        #     self.params.blend_alpha, self.params.output_dir, self.device
        # )
        # 绘制ASR MTA曲线
        plot_training_progress(asr_history, main_acc_history, self.params.output_dir, 
                            self.params.start_round, self.params.end_round, 
                            self.params.poison_start, self.params.poison_end)
        # 绘制L2曲线
        plot_stealth_metrics(self.malicious_l2_dists, self.benign_l2_dists, self.params.output_dir, 
                            self.params.start_round, self.params.end_round)

        # 保存训练指标
        save_training_metrics(
            output_dir=self.params.output_dir,
            start_round=self.params.start_round,
            end_round=self.params.end_round,
            asr_history=asr_history,
            mta_history=main_acc_history,
            malicious_l2=self.malicious_l2_dists,
            benign_l2=self.benign_l2_dists
        )
# ----------------------------------------------------------------------------
# 主函数
# ----------------------------------------------------------------------------

def main():
    """主函数，启动联邦学习实验"""
    params = Params()
    
    # 设置是否加载和保存模型（根据需求修改）
    # 示例：第一阶段，保存模型
    # params.start_round = 1
    # params.end_round = 300
    # params.poison_start = 1  
    # params.poison_end = 200
    # params.load_model = False
    # params.save_model = True
    
    # 示例：第二阶段（加载模型，继续训练）
    params.start_round = 1
    params.end_round = 600
    params.poison_start = 300  
    params.poison_end = 500
    params.load_model = True
    params.load_model_path = "output/2025-05-08_22-30-300/global_model_final.pth"  # 替换为实际路径
    params.save_model = False  # 根据需要设置

    file_path = __file__
    file_name = os.path.basename(file_path)
    params.running_file = file_name
    # 保存到配置项文件
    params.save_config()

    # 设置日志系统
    logger = setup_logging(params.output_dir)
    flag = torch.cuda.is_available()
    logger.info(f"Using {file_name} Starting SATA Backdoor Attack Simulation")
    logger.info(f"cuda is_available = {flag}")


    # 数据加载(对于训练数据增强) 
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=1),          # 数据增强：随机裁剪
        # transforms.RandomHorizontalFlip(p=0.3),        # 数据增强：随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # 划分每个客户端的数据
    train_loaders = []
    if params.use_dirichlet:
        # 使用 Dirichlet Non-IID划分
        client_indices = generate_dirichlet_indices(train_dataset, params.num_clients, alpha=params.dirichlet_alpha)
        for indices in client_indices:
            subset = torch.utils.data.Subset(train_dataset, indices)
            train_loaders.append(torch.utils.data.DataLoader(subset, batch_size=params.batch_size, shuffle=True))
        logger.info(f"Using Dirichlet(alpha={params.dirichlet_alpha}) Distribution division data")
    else:
        # IID 划分：平均分配
        for i in range(params.num_clients):
            subset = torch.utils.data.Subset(
                train_dataset,
                range(i * len(train_dataset) // params.num_clients, (i + 1) * len(train_dataset) // params.num_clients)
            )
            train_loaders.append(torch.utils.data.DataLoader(subset, batch_size=params.batch_size, shuffle=True))
        logger.info("Using IID Distribution division data")

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)
    logger.info(f"Prepared {params.num_clients} client data loaders and test loader")

    # 启动训练
    server = Server(params, logger)
    
    # 根据参数决定是否加载模型
    if params.load_model:
        server.load_model(params.load_model_path)
    
    server.train(train_loaders, test_loader)
    logger.info("Training completed")

if __name__ == "__main__":
    main()