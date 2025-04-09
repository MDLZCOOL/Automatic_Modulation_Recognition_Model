import random
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 参数配置
INPUT_SIZE = 2048  # 输入信号长度（固定）
HIDDEN_SIZE = 2048  # 隐藏层大小（调整为更合理的值）
BATCH_SIZE = 64  # 批量大小
NUM_EPOCHS = 200  # 训练轮数（增加以确保模型充分学习）
LEARNING_RATE = 0.001  # 学习率（保持不变）

# 绘制训练集中前10个样本的实部和虚部数据
def plot_first_10_samples(train_dataset):
    # 获取前10个样本
    samples = [train_dataset[i] for i in range(10)]  # 确保获取的是前10个样本
    
    # 创建一个图形窗口
    plt.figure(figsize=(15, 10))
    
    # 遍历前10个样本
    for idx, sample in enumerate(samples):
        # 检查样本的结构
        if isinstance(sample, tuple) and len(sample) == 2:
            data, label = sample
        else:
            raise ValueError(f"Unexpected sample structure: {sample}")
        
        # 将数据从Tensor转换为numpy数组
        data = data.numpy()
        
        # 分离实部和虚部
        real_part = data[:1024]
        imaginary_part = data[1024:]
        
        # 绘制实部和虚部
        plt.subplot(5, 2, idx + 1)  # 创建子图
        plt.plot(real_part, label='Real Part')
        plt.plot(imaginary_part, label='Imaginary Part')
        plt.title(f'Sample {idx + 1} (Label: {label})')
        plt.legend()
        plt.grid(True)
    
    # 调整子图间距
    plt.tight_layout()
    plt.show()

# 读取文件夹中的所有CSV文件
def read_files_from_folder(folder_path, max_files=30):
    data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                data.append(content)
                if len(data) >= max_files:
                    break
    return data

# 将文件内容按换行符分割为列表
def split_to_list(file_content):
    return [float(x) for x in file_content.split('\n') if x.strip()]  # 去掉空行

# 将数据划分为长度为1024的片段
def split_to_segments(data_list, segment_length=1024):
    segments = [data_list[i:i + segment_length] for i in range(0, len(data_list), segment_length)]
    return segments

# 归一化函数
def normalize_data(data):
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) for x in data]

# 数据预处理函数
def prepare_dataset(dataset_paths):
    all_data = []
    all_noisy_data = []
    
    for folder_path in dataset_paths:
        # 读取并处理数据
        contents = read_files_from_folder(folder_path)
        for content in contents:
            data_list = split_to_list(content)
            real_part, imaginary_part = data_list[:len(data_list)//2], data_list[len(data_list)//2:]

            # # 分别对实部和虚部进行归一化
            # real_part = normalize_data(real_part)
            # imaginary_part = normalize_data(imaginary_part)

            # 将实部和虚部按照相同的规则划分成多个1024长度的片段
            real_segments = split_to_segments(real_part)
            imaginary_segments = split_to_segments(imaginary_part)

            # 合并实部和虚部的片段
            combined_segments = [real + imag for real, imag in zip(real_segments, imaginary_segments)]

            # 给合并数据加上噪声
            for segment in combined_segments:
                noise = np.random.normal(0, 0.05, len(segment))  # 均值为0，标准差为0.05的高斯噪声
                noisy_segment = segment + noise
                all_noisy_data.append(noisy_segment)
                all_data.append(segment)  # 保存无噪声的原始数据作为目标

    # 转换为numpy数组
    X_noisy = np.array(all_noisy_data)
    X_clean = np.array(all_data)

    # 检查数据的形状
    print(f"X_noisy shape: {X_noisy.shape}")  # 应该是 (样本数, 2048)
    print(f"X_clean shape: {X_clean.shape}")  # 应该是 (样本数, 2048)
    
    # 数据集划分
    dataset = TensorDataset(torch.FloatTensor(X_noisy), torch.FloatTensor(X_clean))
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    return train_dataset, test_dataset

# 数据预处理函数（修改版）
def prepare_dataset(dataset_paths):
    all_data = []
    all_noisy_data = []
    
    for folder_path in dataset_paths:
        # 读取并处理数据
        contents = read_files_from_folder(folder_path)
        for content in contents:
            data_list = split_to_list(content)
            real_part, imaginary_part = data_list[:len(data_list)//2], data_list[len(data_list)//2:]

            # 将实部和虚部按照相同的规则划分成多个1024长度的片段
            real_segments = split_to_segments(real_part)
            imaginary_segments = split_to_segments(imaginary_part)

            # 合并实部和虚部的片段
            for real_seg, imag_seg in zip(real_segments, imaginary_segments):
                # 将实部和虚部合并为一个2048长度的向量
                combined_segment = real_seg + imag_seg
                
                # 归一化处理
                combined_segment = normalize_data(combined_segment)
                
                # 给合并数据加上噪声
                noise = np.random.normal(0, 0.05, len(combined_segment))  # 均值为0，标准差为0.05的高斯噪声
                noisy_segment = combined_segment + noise
                
                all_noisy_data.append(noisy_segment)
                all_data.append(combined_segment)  # 保存无噪声的原始数据作为目标

    # 转换为numpy数组
    X_noisy = np.array(all_noisy_data)
    X_clean = np.array(all_data)

    # 检查数据的形状
    print(f"X_noisy shape: {X_noisy.shape}")  # 应该是 (样本数, 2048)
    print(f"X_clean shape: {X_clean.shape}")  # 应该是 (样本数, 2048)
    
    # 数据集划分
    dataset = TensorDataset(torch.FloatTensor(X_noisy), torch.FloatTensor(X_clean))
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    return train_dataset, test_dataset

class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_size=2048):
        super(DenoisingAutoencoder, self).__init__()
        
        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # 解码器部分
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_size),
            nn.Sigmoid()  # 输出范围限制在[0, 1]
            # nn.Tanh()  # 输出范围限制在[-1, 1]
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 训练函数（修改版）
def train_model(model, train_loader, test_loader):
    model = model.to(device)
    criterion = nn.MSELoss()  # 使用均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        # 训练阶段
        model.train()
        running_loss = 0.0
        for noisy_data, clean_data in train_loader:
            noisy_data, clean_data = noisy_data.to(device), clean_data.to(device)
            
            optimizer.zero_grad()
            outputs = model(noisy_data)
            loss = criterion(outputs, clean_data)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * noisy_data.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # 验证阶段
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for noisy_data, clean_data in test_loader:
                noisy_data, clean_data = noisy_data.to(device), clean_data.to(device)
                outputs = model(noisy_data)
                loss = criterion(outputs, clean_data)
                test_loss += loss.item() * noisy_data.size(0)
        
        test_loss /= len(test_loader.dataset)
        print(f'Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {epoch_loss:.6f} | Test Loss: {test_loss:.6f}')
        
        # 保存最佳模型
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), 'denoising_autoencoder.pth')

    print("Training completed.")

def evaluate_model(model, test_loader):
    model.load_state_dict(torch.load('denoising_autoencoder.pth'))
    model.eval()
    model.to(device)
    
    total_mse = 0.0
    with torch.no_grad():
        for noisy_data, clean_data in test_loader:
            noisy_data, clean_data = noisy_data.to(device), clean_data.to(device)
            outputs = model(noisy_data)
            
            # 计算MSE
            mse = mean_squared_error(clean_data.cpu().numpy(), outputs.cpu().numpy())
            total_mse += mse * len(noisy_data)
    
    avg_mse = total_mse / len(test_loader.dataset)
    print(f"Average MSE on test set: {avg_mse:.6f}")
    
    # 可视化一些去噪前后的对比
    plt.figure(figsize=(15, 10))
    for i in range(5):  # 可视化5个样本
        idx = random.randint(0, len(test_dataset)-1)
        noisy_sample, clean_sample = test_dataset[idx]
        
        # 转换为numpy数组
        noisy_sample = noisy_sample.numpy()
        clean_sample = clean_sample.numpy()
        
        # 使用模型去噪
        with torch.no_grad():
            denoised_sample = model(torch.FloatTensor(noisy_sample).unsqueeze(0).to(device)).cpu().numpy().flatten()
        
        # 绘制原始、加噪和去噪后的信号
        plt.subplot(5, 3, i*3 + 1)
        plt.plot(clean_sample)
        plt.title(f"Original Sample {i+1}")
        plt.grid(True)
        
        plt.subplot(5, 3, i*3 + 2)
        plt.plot(noisy_sample)
        plt.title(f"Noisy Sample {i+1}")
        plt.grid(True)
        
        plt.subplot(5, 3, i*3 + 3)
        plt.plot(denoised_sample)
        plt.title(f"Denoised Sample {i+1}")
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 数据准备
    dataset_paths = [
        './dataset_iq/128qam_iq',
        './dataset_iq/16qam_iq',
        './dataset_iq/256qam_iq',
        './dataset_iq/32qam_iq',
        './dataset_iq/64qam_iq',
        './dataset_iq/8psk_iq',
        './dataset_iq/bpsk_iq',
        './dataset_iq/qpsk_iq',
        './dataset_iq/am_iq',
        './dataset_iq/fm_iq'
    ]
    
    train_dataset, test_dataset = prepare_dataset(dataset_paths)

    # 绘制训练集中前10个样本的实部和虚部数据
    plot_first_10_samples(train_dataset)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # 初始化模型
    model = DenoisingAutoencoder(input_size=2048)
    
    # 开始训练
    train_model(model, train_loader, test_loader)


# 评估模型
evaluate_model(model, test_loader)