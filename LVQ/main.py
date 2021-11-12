import numpy as np
import random
from scipy.spatial import Voronoi,voronoi_plot_2d
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
#学习向量量化模型，试图找到一组原型向量来刻画聚类结构，更新迭代原型向量达到聚类目的
class LVQ():
    def __init__(self, q=5, max_iter=300,data=None):
        self.q=q
        self.data=data
        sample_list = [i for i in range(data.shape[0])]
        self.sample_num = random.sample(sample_list, q)
        #print(data[self.sample_num, :])
        #print(np.expand_dims(np.around(np.random.random_sample((q,))),axis=1))
        self.initial = np.append(data[self.sample_num, :-1], np.expand_dims(np.around(np.random.random_sample((q,))),axis=1),axis=1)
#获取原型向量之后采用voronoi剖分
    def update(self,lr=0.5,iter=10):
        dist_array=np.zeros((self.data.shape[0],self.initial.shape[0]))
        clustering=np.zeros((self.data.shape[0],self.data.shape[0]))
        n=self.data.shape[1]
        for it in range(iter):
            for i in range(self.data.shape[0]):
                for j in range(self.initial.shape[0]):
                    dist_array[i][j] = np.sqrt(np.sum(np.square(self.data[i,] - self.initial[j]))) / n
                index = np.argmin(dist_array[i])
                if self.data[i, -1] == self.initial[index, -1]:
                    self.initial[index] = self.initial[index] + lr * (self.data[i] - self.initial[index])
                else:
                    self.initial[index] = self.initial[index] - lr * (self.data[i] - self.initial[index])
        points=self.initial[:,:-1]
        vor=Voronoi(points)
        fig=voronoi_plot_2d(vor)







test=np.loadtxt('b.txt')
label=np.zeros(test.shape[0])
test_label=np.insert(test, 2, label, axis=1)
test_label[9:22,2]=1
print(test_label)
case=LVQ(data=test_label)
#print(case.initial)
case.update(iter=500)
num0 = np.where(test_label[:, -1] == 0)
num1 = np.where(test_label[:, -1] == 1)
logit0 = test[num0]
logit1 = test[num1]
plt.scatter(logit0[:,0],logit0[:,1],c='r')
plt.scatter(logit1[:,0],logit1[:,1],c='y')
plt.show()