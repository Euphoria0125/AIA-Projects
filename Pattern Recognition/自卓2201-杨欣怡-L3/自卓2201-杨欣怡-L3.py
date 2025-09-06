import numpy as np
import matplotlib.pyplot as plt
from time import time


class Dataset:
    def __init__(self, means1=[-5, 0], means2=[0, 5]):
            self._rand_sample(means1, means2)

    def _rand_sample(self, means1, means2):
        means1 = np.array(means1)
        means2 = np.array(means2)
        covar = np.array([1, 0, 0, 1]).reshape(2, 2)
        x1 = np.random.multivariate_normal(means1, covar, size=200)
        x2 = np.random.multivariate_normal(means2, covar, size=200)
        y1 = np.ones((200, 1))
        y2 = np.ones((200, 1)) * -1
        self._split(x1, y1, x2, y2)

    def _split(self, x1, y1, x2, y2):
        num1 = x1.shape[0]
        train_num1 = int(num1 * 0.8)
        num2 = x2.shape[0]
        train_num2 = int(num2 * 0.8)

        self.x_train = np.concatenate((x1[:train_num1], x2[:train_num2]), axis=0)
        self.y_train = np.concatenate((y1[:train_num1], y2[:train_num2]), axis=0)
        self.x_test = np.concatenate((x1[train_num1:], x2[train_num2:]), axis=0)
        self.y_test = np.concatenate((y1[train_num1:], y2[train_num2:]), axis=0)

#各种算法如下
class Adagrad:
    def __init__(self, epsilon=1e-6):
        self.sigma = 0
        self.t = 0
        self.epsilon = epsilon

    def __call__(self, grad, lr):
        self.sigma = np.math.sqrt(((self.sigma ** 2) * self.t + grad ** 2) / (self.t + 1)) + self.epsilon
        self.t = self.t + 1
        return (lr * grad) / self.sigma


class RMSProp(Adagrad):
    def __init__(self, alpha=0.9):
        super(RMSProp, self).__init__()
        self.alpha = alpha

    def __call__(self, grad, lr):
        self.sigma = np.math.sqrt(self.alpha * self.sigma + (1 - self.alpha) * grad ** 2)
        return (lr * grad) / self.sigma


class Momentum:
    def __init__(self, Lambda=0.9):
        self.m = 0
        self.Lambda = Lambda

    def __call__(self, grad, lr):
        self.m = - self.Lambda * self.m + lr * grad
        return self.m


class Adam():
    def __init__(self, beta1=0.9, beta2=0.99, epsilon=1e-6):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0
        self.t = 1

    def __call__(self, grad, lr):
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        if self.t < 5:
            m = self.m / (1 - self.beta1 ** self.t)
            v = self.v / (1 - self.beta2 ** self.t)
        else:
            m = self.m
            v = self.v
        return lr * m / (np.math.sqrt(v) + self.epsilon)


def L2_loss(y_estimated, y):
    num_sample = y.shape[0]
    assert (y_estimated.shape == y.shape)
    loss = (1 / num_sample) * np.sum((y_estimated - y) ** 2)

    return loss


class LinearRegression:
    def __init__(self, lr=None, epoch=None, mode='g'):
        self.mode = mode
        self.w = np.random.randn(3, 1)
        if self.mode == 'g':
            self.lr = lr
            self.epoch = epoch
            assert (self.lr != None)
            assert (self.epoch != None)

    def __call__(self, x, y):

        x = x.reshape(-1, 2)
        y = y.reshape(-1, 1)
        expand_axis = np.ones((x.shape[0], 1))
        x = np.concatenate((expand_axis, x), axis=-1)

        if self.mode == 'a':
            x_gen_inverse = np.matmul(np.linalg.inv(np.matmul(x.T, x)), x.T)
            W = np.matmul(x_gen_inverse, y)

            self.w = W
            y_estimated = np.matmul(x, W)

            return y_estimated

        if self.mode == 'g':
            losses = []
            for i in range(self.epoch):
                y_estimated = np.matmul(x, self.w)
                loss = L2_loss(y_estimated, y)
                losses.append(loss)
                grad = self._calculate_grad(x, y)
                self.w -= self.lr * grad

            return y_estimated, losses

    def _calculate_grad(self, x, y):
        num_sample = x.shape[0]
        aver = (2 / num_sample)
        grad = aver * np.matmul((np.matmul(x, self.w) - y).T, x)
        return grad.T

    def eval(self, x, y):
        test_num = x.shape[0]
        expand_axis = np.ones((x.shape[0], 1))
        x = np.concatenate((expand_axis, x), axis=-1)
        y_estimated = np.matmul(x, self.w)
        y_estimated = np.sign(y_estimated)
        assert (len(y_estimated) == len(y))
        correct_num = len(np.where(y_estimated == y)[0])
        print('分类准确率: %.2f' % (correct_num / test_num))


data2 = Dataset()
x_train = data2.x_train
y_train = data2.y_train
x_test = data2.x_test
y_test = data2.y_test
c1 = plt.scatter(x_train[:80, 0], x_train[:80, 1], alpha=0.5, marker='.', c='green')
c2 = plt.scatter(x_train[80:, 0], x_train[80:, 1], alpha=0.5, marker='^', c='blue')
c3 = plt.scatter(x_test[:20, 0], x_test[:20, 1], alpha=0.5, marker='+', c='green')
c4 = plt.scatter(x_test[20:, 0], x_test[20:, 1], alpha=0.5, marker='x', c='blue')
plt.legend(handles=[c1, c2, c3, c4], labels=['train_x1', 'train_x2', 'test_x1', 'test_x2'], loc='best')
plt.show()

model1 = LinearRegression(mode='a')
y_estimated = model1(x_train, y_train)
model1.eval(x_test, y_test)
w, b = model1.w[1:], model1.w[0]
c1 = plt.scatter(x_train[:80, 0], x_train[:80, 1], alpha=0.5, marker='.', c='green')
c2 = plt.scatter(x_train[80:, 0], x_train[80:, 1], alpha=0.5, marker='^', c='blue')
c3 = plt.scatter(x_test[:20, 0], x_test[:20, 1], alpha=0.5, marker='+', c='green')
c4 = plt.scatter(x_test[20:, 0], x_test[20:, 1], alpha=0.5, marker='x', c='blue')
xmax = np.max(x_train[:, 0])
xmin = np.min(x_train[:, 0])
point1 = [xmin, xmax]
point2 = [-(w[0] * xmin + b) / w[1], -(w[0] * xmax + b) / w[1]]
plt.plot(point1, point2, c='black')
plt.legend(handles=[c1, c2, c3, c4], labels=['train_x1', 'train_x2', 'test_x1', 'test_x2'], loc='best')
plt.title('generalized inverse')
plt.show()

lr = 0.02
epoch = 10

model2 = LinearRegression(lr=lr, epoch=epoch, mode='g')
y_estimated, loss = model2(x_train, y_train)
model2.eval(x_test, y_test)
w, b = model2.w[1:], model2.w[0]
c1 = plt.scatter(x_train[:80, 0], x_train[:80, 1], alpha=0.5, marker='.', c='green')
c2 = plt.scatter(x_train[80:, 0], x_train[80:, 1], alpha=0.5, marker='^', c='blue')
c3 = plt.scatter(x_test[:20, 0], x_test[:20, 1], alpha=0.5, marker='+', c='green')
c4 = plt.scatter(x_test[20:, 0], x_test[20:, 1], alpha=0.4, marker='x', c='blue')
xmax = np.max(x_train[:, 0])
xmin = np.min(x_train[:, 0])
point1 = [xmin, xmax]
point2 = [-(w[0] * xmin + b) / w[1], -(w[0] * xmax + b) / w[1]]
plt.plot(point1, point2, c='black')
plt.legend(handles=[c1, c2, c3, c4], labels=['train_x1',  'train_x2', 'test_x1','test_x2'], loc='best')
plt.title('gradient descent')
plt.show()

epoch = np.arange(0, epoch)
loss = np.array(loss)
plt.plot(epoch, loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

data3 = Dataset(means1=[1, 0], means2=[0, 1])
x_train = data3.x_train
y_train = data3.y_train
x_test = data3.x_test
y_test = data3.y_test
c1 = plt.scatter(x_train[:80, 0], x_train[:80, 1], alpha=0.5, marker='.', c='green')
c2 = plt.scatter(x_train[80:, 0], x_train[80:, 1], alpha=0.5, marker='^', c='blue')
c3 = plt.scatter(x_test[:20, 0], x_test[:20, 1], alpha=0.5, marker='+', c='green')
c4 = plt.scatter(x_test[20:, 0], x_test[20:, 1], alpha=0.5, marker='x', c='blue')
plt.legend(handles=[c1, c2, c3, c4], labels=['train_x1', 'train_x2', 'test_x1', 'test_x2'], loc='best')
plt.show()

model1 = LinearRegression(mode='a')
y_estimated = model1(x_train, y_train)
model1.eval(x_test, y_test)
w, b = model1.w[1:], model1.w[0]
c1 = plt.scatter(x_train[:80, 0], x_train[:80, 1], alpha=0.5, marker='.', c='green')
c2 = plt.scatter(x_train[80:, 0], x_train[80:, 1], alpha=0.5, marker='^', c='blue')
c3 = plt.scatter(x_test[:20, 0], x_test[:20, 1], alpha=0.5, marker='+', c='green')
c4 = plt.scatter(x_test[20:, 0], x_test[20:, 1], alpha=0.5, marker='x', c='blue')
xmax = np.max(x_train[:, 0])
xmin = np.min(x_train[:, 0])
point1 = [xmin, xmax]
point2 = [-(w[0] * xmin + b) / w[1], -(w[0] * xmax + b) / w[1]]
plt.plot(point1, point2, c='black')
plt.legend(handles=[c1, c2, c3, c4], labels=['train_x1',  'train_x2', 'test_x1','test_x2'], loc='best')
plt.title('generalized inverse')
plt.show()

lr = 0.02
epoch = 100

model2 = LinearRegression(lr=lr, epoch=epoch, mode='g')
y_estimated, loss = model2(x_train, y_train)
model2.eval(x_test, y_test)

w, b = model2.w[1:], model2.w[0]

c1 = plt.scatter(x_train[:80, 0], x_train[:80, 1], alpha=0.5, marker='.', c='green')
c2 = plt.scatter(x_train[80:, 0], x_train[80:, 1], alpha=0.5, marker='^', c='blue')
c3 = plt.scatter(x_test[:20, 0], x_test[:20, 1], alpha=0.5, marker='+', c='green')
c4 = plt.scatter(x_test[20:, 0], x_test[20:, 1], alpha=0.5, marker='x', c='blue')
xmax = np.max(x_train[:, 0])
xmin = np.min(x_train[:, 0])
point1 = [xmin, xmax]
point2 = [-(w[0] * xmin + b) / w[1], -(w[0] * xmax + b) / w[1]]
plt.plot(point1, point2, c='black')
plt.legend(handles=[c1, c2, c3, c4], labels=['train_x1',  'train_x2', 'test_x1','test_x2'], loc='best')
plt.title('gradient descent')
plt.show()

epoch = np.arange(0, epoch)
loss = np.array(loss)
plt.plot(epoch, loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


class func:
    def __init__(self):
        pass

    def __call__(self, x):
        y = x * np.cos(0.25 * np.pi * x)
        grad = self._calculate_grad(x)
        return y, grad

    def _calculate_grad(self, x):
        grad = np.cos(0.25 * np.pi * x) - 0.25 * x * np.sin(0.25 * np.pi * x)
        return grad

#迭代十次
epoch = 10
lr = 0.4

f = func()
x = np.ones(4) * -4
y = np.ones(4) * f(-4)[0]
grad = np.zeros(4)
optimizer = [Adagrad(), RMSProp(), Momentum(), Adam()]

xs = []
ys = []
for i in range(epoch):
    for j in range(4):
        y[j], grad[j] = f(x[j])
    xs.append([x[0], x[1], x[2], x[3]])
    ys.append([y[0], y[1], y[2], y[3]])
    for j in range(4):
        x[j] -= optimizer[j](grad[j], lr)
xs = np.array(xs)
ys = np.array(ys)

plt.plot(xs[:, 0], ys[:, 0], c='red', label='Adagrad')
plt.plot(xs[:, 1], ys[:, 1], c='blue', label='RMSProp')
plt.plot(xs[:, 2], ys[:, 2], c='green', label='Momentum')
plt.plot(xs[:, 3], ys[:, 3], c='yellow', label='Adam')

plt.scatter(xs[:, 1::2], ys[:, 1::2], alpha=0.5, c='red', marker='.')
plt.scatter(xs[:, 1::2], ys[:, 1::2], alpha=0.5, c='blue', marker='^')
plt.scatter(xs[:, 1::2], ys[:, 1::2], alpha=0.5, c='green', marker='+')
plt.scatter(xs[:, 1::2], ys[:, 1::2], alpha=0.5, c='yellow', marker='x')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

#迭代50次
epoch = 50
lr = 0.4

f = func()
x = np.ones(4) * -4
y = np.ones(4) * f(-4)[0]
grad = np.zeros(4)
optimizer = [Adagrad(), RMSProp(), Momentum(), Adam(beta1=0.9)]

xs = []
ys = []
for i in range(epoch):
    for j in range(4):
        y[j], grad[j] = f(x[j])
    xs.append([x[0], x[1], x[2], x[3]])
    ys.append([y[0], y[1], y[2], y[3]])
    for j in range(4):
        x[j] -= optimizer[j](grad[j], lr)
xs = np.array(xs)
ys = np.array(ys)

plt.plot(xs[:, 0], ys[:, 0], c='red', label='Adagrad')
plt.plot(xs[:, 1], ys[:, 1], c='blue', label='RMSProp')
plt.plot(xs[:, 2], ys[:, 2], c='green', label='Momentum')
plt.plot(xs[:, 3], ys[:, 3], c='yellow', label='Adam')

plt.scatter(xs[:, 1::5], ys[:, 1::5], alpha=0.5, c='red', marker='.')
plt.scatter(xs[:, 1::5], ys[:, 1::5], alpha=0.5, c='blue', marker='^')
plt.scatter(xs[:, 1::5], ys[:, 1::5], alpha=0.5, c='green', marker='+')
plt.scatter(xs[:, 1::5], ys[:, 1::5], alpha=0.5, c='yellow', marker='x')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
