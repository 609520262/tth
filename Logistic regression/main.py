import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#对率回归模型，实则为二分类问题，采用对率函数得到目标的近似概率，用极大似然法估计参数
class Logistic_regression():
    def __init__(self, data=None):
        self.data=data
        self.Beta=np.ones((self.data.shape[1],1))
        self.data1 = np.append(self.data[:, :-1], np.ones((self.data.shape[0], 1)), axis=1)

#计算1类后验概率
    def p1(self,x):
        return np.exp(x)/(1+np.exp(x))
#计算0类后验概率
    def p0(self,x):
        return 1/ (1 + np.exp(x))
#牛顿法寻找高阶连续凸函数最优解，也可用梯度下降法，两者效果差不多
    def update(self,iter=200,lr=0.01):
        for n in range(iter):
            L=0
            dL=np.zeros((self.data.shape[1],1))
            ddL=np.zeros((self.data1.shape[1],self.data1.shape[1]))
            for i in range(self.data.shape[0]):
                # L = L+(-self.data[i, -1] * np.sum(self.Beta * self.data1[i,:]) + np.log(1 + np.exp(np.sum(self.Beta *self.data1[i,:]))))
                dL=dL+ (self.data1[i,:].reshape(-1,1)*(self.data[i, -1]-self.p1(np.dot(self.Beta.reshape(1,-1),self.data1[i,:].reshape(-1,1)))))
                ddL=ddL+(np.dot(self.data1[i,:].reshape(-1,1),self.data1[i,:].reshape(1,-1))*self.p1(np.dot(self.Beta.reshape(1,-1),self.data1[i,:].reshape(-1,1)))*(1-self.p1(np.dot(self.Beta.reshape(1,-1),self.data1[i,:].reshape(-1,1)))))
            dL=-dL

            ax1.scatter3D(self.Beta[0],self.Beta[1],self.Beta[2])
            self.Beta=self.Beta-np.dot(lr,dL)

#预测函数
    #输入：样本x值
    #输出：样本y值
    def predict(self,x):
        return 1/(1+np.exp(-np.dot(self.Beta.reshape(1,-1),x.reshape(-1,1))))
















test=np.loadtxt('b.txt')
label=np.zeros(test.shape[0])
test_label=np.insert(test, 2, label, axis=1)
test_label[9:22,2]=1
ax1 = plt.axes(projection='3d')

case=Logistic_regression(data=test_label)

case.update()
plt.show()

x=np.array([0.657,0.198,1])
print('预测值为：{}'.format(case.predict(x)))


