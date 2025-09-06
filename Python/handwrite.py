import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 加载手写数字数据集
digits = load_digits()

# 获取特征和标签
X = digits.data
y = digits.target

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 定义神经网络类
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 初始化权重和偏置
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros(self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros(self.output_size)
        
    # 定义前向传播方法
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = np.exp(self.z2) / np.sum(np.exp(self.z2), axis=1, keepdims=True)
        return self.a2
    
    # 定义交叉熵损失函数
    def compute_loss(self, X, y):
        m = y.shape[0]
        probs = self.forward(X)
        loss = -np.log(probs[range(m), y])
        return np.sum(loss) / m
    
    # 定义反向传播方法
    def backward(self, X, y, learning_rate):
        m = y.shape[0]
        deltas = self.forward(X)
        deltas[range(m), y] -= 1
        deltas /= m
        
        dW2 = np.dot(self.a1.T, deltas)
        db2 = np.sum(deltas, axis=0)
        dW1 = np.dot(X.T, np.dot(deltas, self.W2.T) * (1 - np.power(self.a1, 2)))
        db1 = np.sum(np.dot(deltas, self.W2.T) * (1 - np.power(self.a1, 2)), axis=0)
        
        # 更新权重和偏置
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    # 定义模型训练方法
    def train(self, X, y, learning_rate, num_epochs):
        self.loss_history = []
        
        for epoch in range(num_epochs):
            self.backward(X, y, learning_rate)
            loss = self.compute_loss(X, y)
            self.loss_history.append(loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")
    
    # 定义模型预测方法
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# 定义模型参数
input_size = X_train.shape[1]
hidden_size = 64
output_size = len(np.unique(y_train))

# 创建神经网络模型
model = NeuralNetwork(input_size, hidden_size, output_size)

# 训练模型
learning_rate = 0.1
num_epochs = 100
model.train(X_train, y_train, learning_rate, num_epochs)

# 绘制损失曲线
plt.plot(range(num_epochs), model.loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# 在测试集上评估模型
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy:.4f}")
