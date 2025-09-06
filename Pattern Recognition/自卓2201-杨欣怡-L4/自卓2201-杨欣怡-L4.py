import numpy as np
import matplotlib.pyplot as plt


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


class Fisher:
    def __init__(self):
        self.w = np.zeros((2, 1))
        self.thresh = 0

    def train(self, x, y):
        assert (x.shape[0] == y.shape[0])
        y = y.flatten()

        cls1_idx = np.where(y == 1)
        cls2_idx = np.where(y == -1)
        c1 = x[cls1_idx].T
        c2 = x[cls2_idx].T

        mean1 = np.mean(c1, axis=-1)
        mean2 = np.mean(c2, axis=-1)
        covar1 = np.cov(c1)
        covar2 = np.cov(c2)

        sw = covar1 + covar2
        self.w = np.matmul(np.linalg.inv(sw), (mean1 - mean2))
        self.thresh = np.matmul(self.w.T, mean1 + mean2) / 2
        print('thereshold: {}'.format(self.thresh))

        return self.w, self.thresh

    def eval(self, x, y):
        assert (x.shape[0] == y.shape[0])

        y_estimated = np.matmul(x, self.w)
        y_estimated = np.sign(y_estimated - self.thresh)
        y_estimated = y_estimated.reshape(-1, 1)
        correct_num = len(np.where((y_estimated - y) == 0)[1])
        accuracy = correct_num / y.shape[0]
        print('分类准确率: %.2f' % accuracy)


dataset = Dataset()
x_train = dataset.x_train
y_train = dataset.y_train
x_test = dataset.x_test
y_test = dataset.y_test

c1 = plt.scatter(x_train[:160, 0], x_train[:160, 1], alpha=0.5, marker='.', c='green')
c2 = plt.scatter(x_train[160:, 0], x_train[160:, 1], alpha=0.5, marker='^', c='blue')
c3 = plt.scatter(x_test[:40, 0], x_test[:40, 1], alpha=0.5, marker='+', c='green')
c4 = plt.scatter(x_test[40:, 0], x_test[40:, 1], alpha=0.5, marker='x', c='blue')
plt.legend(handles=[c1, c2, c3, c4], labels=['train_x1', 'train_x2', 'test_x1', 'test_x2'], loc='best')
plt.show()

classifier = Fisher()
w, thresh = classifier.train(x_train, y_train)
classifier.eval(x_train, y_train)

classifier.eval(x_test, y_test)

xmin = np.min(x_train[:, 0])
xmax = np.max(x_train[:, 0])
center = np.mean(x_train, axis=0)
k = w[1] / w[0]
b = center[1] - k * center[0]
print(center)
print(k)
print(b)

point1 = [xmin, xmax]
point2 = [k * xmin + b, k * xmax + b]

c1 = plt.scatter(x_train[:160, 0], x_train[:160, 1], alpha=0.5, marker='.', c='green')
c2 = plt.scatter(x_train[160:, 0], x_train[160:, 1], alpha=0.5, marker='^', c='blue')
c3 = plt.scatter(x_test[:40, 0], x_test[:40, 1], alpha=0.5, marker='+', c='green')
c4 = plt.scatter(x_test[40:, 0], x_test[40:, 1], alpha=0.5, marker='x', c='blue')
x1 = ((k * center[1] - b) + center[0]) / (1 + k ** 2)
thresh_point = [x1, k * x1 + b]
thresh = plt.scatter(thresh_point[0], thresh_point[1], marker='s', c='yellow', linewidths=3)
plt.plot(point1, point2, c='black')
plt.legend(handles=[c1, c2, c3, c4, thresh], labels=['train_x1', 'train_x2', 'test_x1', 'test_x2', 'threshold'], loc='best')
plt.show()
