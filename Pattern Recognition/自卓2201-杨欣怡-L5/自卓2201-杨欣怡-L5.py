import numpy as np
import matplotlib.pyplot as plt
from time import time


#定义数据类
class Dataset:
    def __init__(self, means1=[-5, 0], means2=[0, 5]):
        self._rand_sample(means1, means2)

    def _rand_sample(self, means1, means2):
        means1 = np.array(means1)
        means2 = np.array(means2)
        covar = np.array([1, 0, 0, 1]).reshape(2,2)
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

        self.x_train = np.concatenate((x1[:train_num1],x2[:train_num2]),axis=0)
        self.y_train = np.concatenate((y1[:train_num1],y2[:train_num2]),axis=0)
        self.x_test = np.concatenate((x1[train_num1:],x2[train_num2:]),axis=0)
        self.y_test = np.concatenate((y1[train_num1:],y2[train_num2:]),axis=0)

        
#定义logistics回归模型
import numpy as np
        
class LogisticRegression:
    def __init__(self, lr=0.02, epoch=50):
        '''
        mode: g: Grandient Descent
              a: analysis
        '''
        self.w = np.zeros((3,1))
        self.lr = lr
        self.epoch = epoch

    def __call__(self, x, y):

        x = x.reshape(-1,2)
        y = y.reshape(-1,1)
        expand_axis = np.ones((x.shape[0],1))
        x = np.concatenate((expand_axis, x), axis=-1)

     
        losses = []
        for i in range(self.epoch):
            yhat = self.sigmoid(np.matmul(x, self.w))
            loss = self.cross_entropy(x, y, yhat)
            losses.append(loss)
            grad = self._calculate_grad(x, y, yhat)
        
            self.w -= self.lr * grad
            print('epoch: %i/%i loss: %.2f'%(i+1, self.epoch, loss))

        return yhat, losses

    def _calculate_grad(self, x, y, yhat):
        batchsize = y.shape[0]
        s = np.matmul(x, self.w)
        grad = (self.sigmoid(-y*s) - 1) * y * x
        grad = np.sum(grad, axis=0) / batchsize
        return grad.reshape(3,1)

    def cross_entropy(self,x, y, yhat):
        num = y.shape[0]
        s = np.matmul(x, self.w)
        loss = (1/num)*np.sum(np.log(1+np.exp(-s*y)))
        return loss

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def eval(self, x, y):
        test_num = x.shape[0]
        expand_axis = np.ones((x.shape[0],1))
        x = np.concatenate((expand_axis, x), axis=-1)
        yhat = np.matmul(x, self.w)
        yhat = np.sign(self.sigmoid(yhat) - 0.5)
        assert(len(yhat) == len(y))
        correct_num = len(np.where(yhat == y)[0])
        print('accuracy: %.2f'%(correct_num/test_num))
        
#加载数据集
data2 = Dataset()
x_train = data2.x_train
y_train = data2.y_train
x_test = data2.x_test
y_test = data2.y_test
print("x_train: {}".format(x_train.shape))
print("y_train: {}".format(y_train.shape))
print("x_test: {}".format(x_test.shape))
print("y_test: {}".format(y_test.shape))
c1 = plt.scatter(x_train[:160,0], x_train[:160,1], alpha=0.6, marker='.', c='green')
c2 = plt.scatter(x_train[160:,0], x_train[160:,1], alpha=0.6, marker='^', c='blue')
c3 = plt.scatter(x_test[:40,0], x_test[:40,1], alpha=0.6, marker='+', c='green')
c4 = plt.scatter(x_test[40:,0], x_test[40:,1], alpha=0.6, marker='x', c='blue')
plt.legend(handles=[c1, c2, c3, c4],labels=['train_x1','train_x2', 'test_x1', 'test_x2'],loc='best')
plt.show()

epoch = 50
lr = 0.01
model = LogisticRegression(lr=lr, epoch=epoch)
yhat, loss = model(x_train, y_train)

w, b = model.w[1:], model.w[0]
print('w: {}'.format(w))
print('b: {}'.format(b))
c1 = plt.scatter(x_train[:160,0], x_train[:160,1], alpha=0.4, marker='.', c='green')
c2 = plt.scatter(x_train[160:,0], x_train[160:,1], alpha=0.4, marker='^', c='blue')
c3 = plt.scatter(x_test[:40,0], x_test[:40,1], alpha=0.4, marker='+', c='green')
c4 = plt.scatter(x_test[40:,0], x_test[40:,1], alpha=0.4, marker='x', c='blue')
xmax = np.max(x_train[:,0])
xmin = np.min(x_train[:,0])
point1 = [xmin,xmax]
point2 = [-(w[0]*xmin+b)/w[1],-(w[0]*xmax+b)/w[1]]
plt.plot(point1,point2,c='black')
plt.legend(handles=[c1, c3, c2, c4],labels=['train_x1','train_x2', 'test_x1', 'test_x2'],loc='best')
plt.title('logistic regression')
plt.show()

epochs = np.arange(0,epoch)
loss = np.array(loss)
plt.plot(epochs,loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
#计算准确率
model.eval(x_test,y_test)
