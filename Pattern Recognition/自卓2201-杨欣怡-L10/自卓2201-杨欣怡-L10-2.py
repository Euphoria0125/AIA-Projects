import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# 加载和预处理数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.expand_dims(x_train, axis=-1)  # (60000, 28, 28, 1)
x_test = np.expand_dims(x_test, axis=-1)    # (10000, 28, 28, 1)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 构建LeNet模型
def build_lenet_model():
    model = models.Sequential()
    model.add(layers.Conv2D(6, (5, 5), strides=1, padding='same', activation='sigmoid', input_shape=(28, 28, 1)))
    model.add(layers.AvgPool2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(16, (5, 5), strides=1, padding='valid', activation='sigmoid'))
    model.add(layers.AvgPool2D(pool_size=(2, 2), strides=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation='sigmoid'))
    model.add(layers.Dense(84, activation='sigmoid'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

# 编译和训练模型
model = build_lenet_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, batch_size=256, validation_data=(x_test, y_test))

# 绘制训练曲线
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Test Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

plot_training_history(history)

# 随机抽取10个测试集样本并显示预测结果
random_indices = np.random.randint(0, x_test.shape[0], 10)
x_random_samples = x_test[random_indices]
y_random_samples = y_test[random_indices]
y_pred = model.predict(x_random_samples)

plt.figure(figsize=(12, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_random_samples[i].reshape(28, 28), cmap='gray')
    plt.title(f"True: {np.argmax(y_random_samples[i])}, Pred: {np.argmax(y_pred[i])}")
    plt.axis('off')
plt.show()
