import random
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split


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
def read_files_from_folder(folder_path):
    data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                data.append(content)
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

# 将实部和虚部堆叠成一个2048长度的输出
def stack_real_imaginary(real_part, imaginary_part):
    return real_part.extend(imaginary_part)

# 数据预处理函数
def prepare_dataset(dataset_paths, keys):
    all_data = []
    all_labels = []
    
    for idx, folder_path in enumerate(dataset_paths):
        # 读取并处理数据
        contents = read_files_from_folder(folder_path)
        # print(type(contents[0]))
        for content in contents:
            data_list = split_to_list(content)
            real_part, imaginary_part = data_list[:len(data_list)//2], data_list[len(data_list)//2:]

            # 分别对实部和虚部进行归一化
            real_part = normalize_data(real_part)
            imaginary_part = normalize_data(imaginary_part)

            # 将实部和虚部按照相同的规则划分成多个1024长度的片段
            real_segments = split_to_segments(real_part)
            imaginary_segments = split_to_segments(imaginary_part)

            # 合并实部和虚部的片段
            combined_segments = [real + imag for real, imag in zip(real_segments, imaginary_segments)]

            # 给合并数据加上噪声
            for segment in combined_segments:
                noise = np.random.normal(0, 0.1, len(segment))  # 均值为0，标准差为0.1的高斯噪声
                segment += noise

            # 归一化合并后的片段
            for segment in combined_segments:
                segment = normalize_data(segment)

            # 添加数据和标签
            all_data.extend(combined_segments)
            all_labels.extend([idx] * len(combined_segments))  # 使用索引作为类别标签

      
    # 转换为numpy数组
    X = np.array(all_data)
    y = np.array(all_labels)

    # 检查数据和标签的形状
    print(f"X shape: {X.shape}")  # 应该是 (样本数, 1024)
    print(f"y shape: {y.shape}")  # 应该是 (样本数,)
    
    # 数据集划分
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    return train_dataset, test_dataset

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 参数配置
INPUT_SIZE = 2048
HIDDEN_SIZE = 512
NUM_CLASSES = 8
BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 0.001

# 模型结构
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.bn1 = nn.BatchNorm1d(HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE//2)
        self.bn2 = nn.BatchNorm1d(HIDDEN_SIZE//2)
        self.fc3 = nn.Linear(HIDDEN_SIZE//2, NUM_CLASSES)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        return self.fc3(x)

# 训练函数
def train_model(model, train_loader, test_loader):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        # 训练阶段
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # 验证阶段
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        acc = correct / total
        print(f'Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {epoch_loss:.4f} | Acc: {acc:.4f}')
        
        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'model.pth')
    
    # 最终评估
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=keys)
    disp.plot()
    plt.show()

if __name__ == "__main__":
    # 数据准备
    dataset_paths = [
        './dataset/128qam_iq',
        './dataset/16qam_iq',
        './dataset/256qam_iq',
        './dataset/32qam_iq',
        './dataset/64qam_iq',
        './dataset/8psk_iq',
        './dataset/bpsk_iq',
        './dataset/qpsk_iq'
    ]
    keys = ['128qam', '16qam', '256qam', '32qam', '64qam', '8psk', 'bpsk', 'qpsk']
    
    train_dataset, test_dataset = prepare_dataset(dataset_paths, keys)

    # 绘制训练集中前10个样本的实部和虚部数据
    plot_first_10_samples(train_dataset)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # 初始化模型
    model = Model()
    
    # 开始训练
    train_model(model, train_loader, test_loader)
