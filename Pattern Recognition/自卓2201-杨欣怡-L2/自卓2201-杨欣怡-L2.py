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


#PLA算法
class PLA(object):
    def __init__(self, dimension):
        super(PLA, self).__init__()
        self.dimension = dimension
        self.W = np.zeros((1, dimension))
        self.b = 0

    def update_param(self, x, y):
        self.W += x * y
        self.b += y

    def train(self, x, y):
        num, dim = x.shape
        if dim != self.dimension:
            raise
        optimized = False
        while not optimized:
            for i in range(num):
                y_estimated = np.dot(self.W, x[i]) + self.b
                if y_estimated * y[i] <= 0:
                    self.update_param(x[i], y[i])
                    break
                if i == num - 1:
                    optimized = True
        print('overtraining')

    def inference(self, x, y):
        num, dim = x.shape
        if dim != self.dimension:
            raise
        y_estimated = np.matmul(self.W, x.transpose(1, 0)) + self.b
        y_estimated = np.sign(y_estimated).transpose(1, 0)
        accuracy = 1 - len(np.nonzero(y_estimated - y)[0]) / len(y)
        print('PLA的分类正确率为: %.2f' % accuracy)

        return y_estimated


class Pocket(object):
    def __init__(self, dimension):
        super(Pocket, self).__init__()
        self.dimension = dimension
        self.W = np.zeros((1, dimension))
        self.b = 0

    def _error_eval(self, w, b, x, y):
        y_estimated = np.matmul(w, x.transpose(1, 0)) + b
        y_estimated = np.sign(y_estimated).transpose(1, 0)
        a = np.nonzero(y_estimated - y)
        error_idxs = np.nonzero(y_estimated - y)[0]

        return len(error_idxs), error_idxs

    def train(self, x, y):
        w = np.random.randn(1, 2)
        b = np.random.randn(1)
        for i in range(1000):
            error_num, error_idxs = self._error_eval(w, b, x, y)
            if error_num == 0:
                break
            else:
                error_idx = np.random.choice(error_idxs)
                error_x = x[error_idx]
                error_y = y[error_idx]
                w_new = w + error_y * error_x
                b_new = b + error_y
                error_num_new, error_idxs_new = self._error_eval(w_new, b_new, x, y)
                if error_num_new <= error_num:
                    error_num = error_num_new
                    error_idxs = error_idxs_new
                    w = w_new
                    b = b_new
        self.W = w
        self.b = b

    def inference(self, x, y):
        num, dim = x.shape
        if dim != self.dimension:
            raise
        y_estimated = np.matmul(self.W, x.transpose(1, 0)) + self.b
        y_estimated = np.sign(y_estimated).transpose(1, 0)
        accuracy = 1 - len(np.nonzero(y_estimated - y)[0]) / len(y)
        print('Pocket的分类正确率: %.2f' % accuracy)

        return y_estimated


data2 = Dataset()
x_train = data2.x_train
y_train = data2.y_train
x_test = data2.x_test
y_test = data2.y_test
c1 = plt.scatter(x_train[:160, 0], x_train[:160, 1], alpha=0.5, marker='^', c='green')
c2 = plt.scatter(x_train[160:, 0], x_train[160:, 1], alpha=0.5, marker='.', c='yellow')
c3 = plt.scatter(x_test[:40, 0], x_test[:40, 1], alpha=0.5, marker='x', c='green')
c4 = plt.scatter(x_test[40:, 0], x_test[40:, 1], alpha=0.5, marker='+', c='yellow')
plt.legend(handles=[c1, c2, c3, c4], labels=['train_x1', 'train_x2', 'test_x1', 'test_x2'], loc='best')
plt.show()

classifier = PLA(dimension=2)
start_time = time()
classifier.train(x_train, y_train)
end_time = time()
print('PLA运行时间: %.3f' % (end_time - start_time))

y_estimated = classifier.inference(x_test, y_test)
w = classifier.W.squeeze()
b = classifier.b

plt.title('PLA')
plt.scatter(x_train[:160, 0], x_train[:160, 1], alpha=0.5, marker='^', c='green')
plt.scatter(x_train[160:, 0], x_train[160:, 1], alpha=0.5, marker='.', c='yellow')
plt.scatter(x_test[:40, 0], x_test[:40, 1], alpha=0.5, marker='x', c='green')
plt.scatter(x_test[40:, 0], x_test[40:, 1], alpha=0.5, marker='+', c='yellow')

xmax = np.max(x_train[:, 0])
xmin = np.min(x_train[:, 0])
point1 = [xmin, xmax]
point2 = [-(w[0] * xmin + b) / w[1], -(w[0] * xmax + b) / w[1]]
plt.plot(point1, point2, c='black')
plt.legend(handles=[c1, c2, c3, c4], labels=['train_x1', 'train_x2', 'test_x1', 'test_x2'], loc='best')
plt.title('PLA')
plt.show()

classifier = Pocket(dimension=2)
start_time = time()
classifier.train(x_train, y_train)
end_time = time()
print('Pocket运行时间: %.3f' % (end_time - start_time))

y_estimated = classifier.inference(x_test, y_test)
w = classifier.W.squeeze()
b = classifier.b

plt.scatter(x_train[:160, 0], x_train[:160, 1], alpha=0.5, marker='^', c='green')
plt.scatter(x_train[160:, 0], x_train[160:, 1], alpha=0.5, marker='.', c='yellow')
plt.scatter(x_test[:40, 0], x_test[:40, 1], alpha=0.5, marker='x', c='green')
plt.scatter(x_test[40:, 0], x_test[40:, 1], alpha=0.5, marker='+', c='yellow')

xmax = np.max(x_train[:, 0])
xmin = np.min(x_train[:, 0])
point1 = [xmin, xmax]
point2 = [-(w[0] * xmin + b) / w[1], -(w[0] * xmax + b) / w[1]]
plt.plot(point1, point2, c='black')
plt.legend(handles=[c1, c2, c3, c4], labels=['train_x1', 'train_x2', 'test_x1', 'test_x2'], loc='best')
plt.title('Pocket')
plt.show()

data2 = Dataset(means1=[1, 0], means2=[0, 1])
x_train = data2.x_train
y_train = data2.y_train
x_test = data2.x_test
y_test = data2.y_test
c1 = plt.scatter(x_train[:160, 0], x_train[:160, 1], alpha=0.5, marker='.', c='green')
c2 = plt.scatter(x_train[160:, 0], x_train[160:, 1], alpha=0.5, marker='^', c='yellow')
c3 = plt.scatter(x_test[:40, 0], x_test[:40, 1], alpha=0.5, marker='+', c='green')
c4 = plt.scatter(x_test[40:, 0], x_test[40:, 1], alpha=0.6, marker='x', c='yellow')
plt.legend(handles=[c1, c2, c3, c4], labels=['train_x1', 'train_x2', 'test_x1', 'test_x2'], loc='best')
plt.show()

classifier = Pocket(dimension=2)
start_time = time()
classifier.train(x_train, y_train)
end_time = time()
print('Pocket运行时间: %.3f' % (end_time - start_time))

y_estimated = classifier.inference(x_test, y_test)
w = classifier.W.squeeze()
b = classifier.b

plt.scatter(x_train[:160, 0], x_train[:160, 1], alpha=0.5, marker='.', c='green')
plt.scatter(x_train[160:, 0], x_train[160:, 1], alpha=0.5, marker='^', c='yellow')
plt.scatter(x_test[:40, 0], x_test[:40, 1], alpha=0.5, marker='+', c='green')
plt.scatter(x_test[40:, 0], x_test[40:, 1], alpha=0.5, marker='x', c='yellow')

xmax = np.max(x_train[:, 0])
xmin = np.min(x_train[:, 0])
point1 = [xmin, xmax]
point2 = [-(w[0] * xmin + b) / w[1], -(w[0] * xmax + b) / w[1]]
plt.plot(point1, point2, c='black')
plt.legend(handles=[c1, c2, c3, c4], labels=['train_x1', 'train_x2', 'test_x1', 'test_x2'], loc='best')
plt.title('Pocket')
plt.show()
