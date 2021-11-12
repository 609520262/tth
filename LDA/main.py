import numpy as np
import matplotlib.pyplot as plt
#线性判别分析，用作二分类问题，后期将延伸到多分类问题
#self.W包含投影直线方向向量信息
class LDA():
    def __init__(self, data=None):
        self.data=data
        self.data=data[np.argsort(data[:,-1])]
        self.W=np.zeros((data.shape[1]-1,1))
#核心在于构造LDA最优化目标
    #SW为类内散度矩阵
    def update(self):
        SW=np.zeros((self.data.shape[1]-1,self.data.shape[1]-1))
        ui=np.zeros((2,1,self.data.shape[1]-1))
        for i in range(2):
            num = np.where(self.data[:, -1] == i)
            print('第{}轮num：{}'.format(i,num))
            logit = self.data[num,:-1]
            logit=np.squeeze(logit)
            ui[i]=np.average(logit,axis=0)
            print('第{}轮ui：{}'.format(i,ui))
            for j in range(logit.shape[0]):
                SW=SW+np.dot(logit[j,:].reshape(-1,1)-ui[i].reshape(-1,1),logit[j,:].reshape(1,-1)-ui[i].reshape(1,-1))
            print('把第{}类计算出来了：{}'.format(i,SW))
        self.W=np.dot(np.linalg.inv(SW),(ui[0]-ui[i]).reshape(2,1))

        self.data_new=np.array([np.dot(self.W.T,self.data[i,:-1].T) for i in range(self.data.shape[0])])
        print(self.data_new)

























test = np.loadtxt('b.txt')
label = np.zeros(test.shape[0])
test_label = np.insert(test, 2, label, axis=1)
test_label[9:21, 2] = 1
case=LDA(data=test_label)
print(case.data)
case.update()
plt.scatter(case.data[:,0],case.data[:,1],c=case.data[:,2],cmap="viridis")
plt.plot([0,case.W[0]],[0,case.W[1]])

plt.show()
