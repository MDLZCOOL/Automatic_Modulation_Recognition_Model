import torch
import numpy as np
from torch import nn

class ModulationClassifier:
    def __init__(self, model_path='best_model_a.pth', device='auto'):
        """
        调制分类器初始化
        :param model_path: 模型文件路径（默认'best_model.pth'）
        :param device: 指定设备（'cuda', 'cpu'）或自动选择（'auto'）
        """
        # 设备配置
        self.device = torch.device(device) if device != 'auto' else \
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 定义与训练时一致的模型结构
        class TrainedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(1024, 512)
                self.bn1 = nn.BatchNorm1d(512)
                self.fc2 = nn.Linear(512, 256)
                self.bn2 = nn.BatchNorm1d(256)
                self.fc3 = nn.Linear(256, 8)
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

        # 初始化并加载模型
        self.model = TrainedModel().to(self.device)
        self.load_model(model_path)
        
        # 类别标签映射
        self.classes = [
            '128qam', '16qam', '256qam', '32qam',
            '64qam', '8psk', 'bpsk', 'qpsk'
        ]

    def load_model(self, model_path):
        """加载预训练权重"""
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()  # 切换到评估模式
            print(f"成功加载模型来自: {model_path}")
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")

    def _validate_input(self, input_data):
        """输入数据验证"""
        if not isinstance(input_data, (list, np.ndarray, torch.Tensor)):
            raise TypeError("输入必须是list、numpy数组或PyTorch张量")
            
        if len(input_data) != 1024:
            raise ValueError(f"输入长度必须为1024，当前长度: {len(input_data)}")
            
        if isinstance(input_data, list):
            if not all(isinstance(x, (int, float)) for x in input_data):
                raise ValueError("列表包含非数值类型元素")

    def predict(self, input_data, return_prob=False):
        """
        执行预测
        :param input_data: 输入数据（1024长度）
        :param return_prob: 是否返回概率分布
        :return: 预测结果（类别或概率）
        """
        # 输入验证
        self._validate_input(input_data)
        
        # 转换为Tensor
        if not isinstance(input_data, torch.Tensor):
            tensor_input = torch.tensor(input_data, dtype=torch.float32)
        else:
            tensor_input = input_data.to(torch.float32)
        
        # 设备转移和维度调整
        tensor_input = tensor_input.to(self.device).unsqueeze(0)  # 添加batch维度
        
        # 执行推理
        with torch.no_grad():
            output = self.model(tensor_input)
            probabilities = torch.softmax(output, dim=1)
        
        # 结果处理
        if return_prob:
            return probabilities.cpu().numpy()[0]  # 返回numpy数组
        else:
            class_index = torch.argmax(probabilities).item()
            return self.classes[class_index]

    def batch_predict(self, batch_data):
        """
        批量预测
        :param batch_data: 形状为[N, 1024]的数组
        :return: 预测结果列表
        """
        tensor_batch = torch.tensor(batch_data, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor_batch)
            probs = torch.softmax(outputs, dim=1)
            indices = torch.argmax(probs, dim=1)
        
        return [self.classes[idx.item()] for idx in indices]

# 示例用法
if __name__ == "__main__":
    # 初始化分类器
    classifier = ModulationClassifier()
    
    # 生成测试数据（示例）
    test_data = np.random.randn(1024)  # 随机生成1024长度的数据
    
    # 单样本预测
    try:
        # 获取类别预测
        prediction = classifier.predict(test_data)
        print(f"预测类别: {prediction}")
        
        # 获取概率分布
        probabilities = classifier.predict(test_data, return_prob=True)
        print(f"概率分布: {probabilities}")
    except Exception as e:
        print(f"预测出错: {str(e)}")
    
    # 批量预测示例
    batch_test = np.random.randn(5, 1024)  # 5个样本
    batch_results = classifier.batch_predict(batch_test)
    print(f"\n批量预测结果: {batch_results}")
