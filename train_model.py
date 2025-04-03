import random
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

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

# 将文件里面的以逗号分隔的浮点数转换为列表
def convert_to_list(file_content):
    return [float(x) for x in file_content.split(',')]

# 将列表以 1024 个元素为一组进行分组
def group_list(lst, group_size=1024):
    return [lst[i:i+group_size] for i in range(0, len(lst), group_size)]

# 将分组后的列表划分为训练集和测试集，训练集占 80%，测试集占 20%（ 要求随机划分 ）
def split_train_test(lst, train_ratio=0.8):
    random.shuffle(lst)
    split_index = int(len(lst) * train_ratio)
    return lst[:split_index], lst[split_index:]

# 判断测试集是否为训练集的子集
def is_subset(list1, list2):
    set1 = set(tuple(lst) for lst in list1)
    set2 = set(tuple(lst) for lst in list2)
    return set2.issubset(set1)

# 将训练集和测试集直接转化为 numpy 数组
def convert_to_numpy(train_list, test_list):
    return np.array(train_list), np.array(test_list)

# 遍历一个文件夹，以一个列表输出所有文件名
def get_file_name(file_path):
    file_list = []
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if file.endswith('.dat'):
                file_list.append(os.path.join(root, file))
    return file_list

# 读取npy文件，并将其转换为列表
def read_npy(file_path):
    return np.load(file_path).tolist()

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 参数配置
INPUT_SIZE = 1024
HIDDEN_SIZE = 512
NUM_CLASSES = 8
BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 0.001

# 数据预处理函数
def prepare_dataset(dataset_paths, keys):
    all_data = []
    all_labels = []
    
    for idx, folder_path in enumerate(dataset_paths):
        # 读取并处理数据
        contents = read_files_from_folder(folder_path)
        for content in contents:
            data_list = convert_to_list(content)
            grouped = group_list(data_list)
            
            # 添加数据和标签
            all_data.extend(grouped)
            all_labels.extend([idx] * len(grouped))  # 使用索引作为类别标签
    
    # 转换为numpy数组
    X = np.array(all_data)
    y = np.array(all_labels)
    
    # 数据集划分
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    return train_dataset, test_dataset

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
        if acc > ——_acc:
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
        './dataset/128qam',
        './dataset/16qam',
        './dataset/256qam',
        './dataset/32qam',
        './dataset/64qam',
        './dataset/8psk',
        './dataset/bpsk',
        './dataset/qpsk'
    ]
    keys = ['128qam', '16qam', '256qam', '32qam', '64qam', '8psk', 'bpsk', 'qpsk']
    
    train_dataset, test_dataset = prepare_dataset(dataset_paths, keys)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # 初始化模型
    model = Model()
    
    # 开始训练
    train_model(model, train_loader, test_loader)
