import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 加载IRIS数据集
iris = datasets.load_iris()
X = iris.data  # 特征数据
y = iris.target  # 标签数据

# 将标签转换为one-hot编码
y = tf.keras.utils.to_categorical(y, num_classes=3)

# 将数据集分为训练集和测试集
# 每个类别随机选30个样本作为训练数据，剩余20个样本作为测试数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)

def create_model(hidden_layers=[64], activation='relu', learning_rate=0.001):
    model = tf.keras.Sequential()
    
    # 输入层
    model.add(tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)))
    
    # 隐含层
    for units in hidden_layers:
        model.add(tf.keras.layers.Dense(units, activation=activation))
    
    # 输出层
    model.add(tf.keras.layers.Dense(3, activation='softmax'))  # softmax用于多分类
    
    # 编译模型
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# 设置不同的网络参数组合进行实验
hidden_layers_configs = [
    [64],              # 1个隐含层，64个节点
    [32, 32],          # 2个隐含层，每层32个节点
    [128, 64],         # 2个隐含层，128个和64个节点
    [128, 64, 32],     # 3个隐含层，128、64和32个节点
]

activation_functions = ['relu', 'sigmoid', 'tanh']
learning_rates = [0.01, 0.001, 0.0001]

# 训练和评估所有配置组合
results = []

for hidden_layers in hidden_layers_configs:
    for activation in activation_functions:
        for lr in learning_rates:
            print(f"Training with hidden_layers={hidden_layers}, activation={activation}, learning_rate={lr}")
            
            # 创建模型
            model = create_model(hidden_layers=hidden_layers, activation=activation, learning_rate=lr)
            
            # 训练模型
            history = model.fit(X_train, y_train, epochs=50, batch_size=10, validation_data=(X_test, y_test), verbose=0)
            
            # 评估模型
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
            
            # 记录结果
            results.append({
                'hidden_layers': hidden_layers,
                'activation': activation,
                'learning_rate': lr,
                'accuracy': accuracy
            })

# 输出实验结果
results_df = pd.DataFrame(results)
print(results_df)

# 绘制训练过程的损失和准确率曲线
def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    
    # 绘制训练和验证的损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制训练和验证的准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

plot_training_history(history)