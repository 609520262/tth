import numpy as np
import random
import math
import matplotlib.pyplot as plt
#高斯混合聚类模型，先根据当前参数计算每个样本属于高斯成分的后验概率，后更新参数

class Mixture_of_Gaussian():
#初始化均值向量、协方差矩阵、混合系数
    def __init__(self,mixture_coefficent_initial=1/3, k=3,data=None):
        self.k=k
        self.data=data
        self.mixture_coefficient=np.array([mixture_coefficent_initial]*k)
        sample_list = [i for i in range(data.shape[0])]
        self.sample_num = random.sample(sample_list, k)
        self.mui=data[self.sample_num]
        cov=np.random.random(size=(data.shape[1],data.shape[1]))
        cov=np.array([[0.1,0.0],[0.0,0.1]])
        self.cov=np.expand_dims(cov, 0).repeat(k, axis=0)
#采用极大似然法更新迭代参数
    def update(self,iter=5):
        Pm=np.zeros((self.data.shape[0],self.k))
        for n in range(iter):
            for i in range(self.data.shape[0]):
                for j in range(self.k):

                    exp=math.exp(-0.5*np.dot(np.dot(np.transpose(self.data[i]-self.mui[j]),np.linalg.inv(self.cov[j])),self.data[i]-self.mui[j]))

                    temp=0
                    for l in range(self.k):
                        temp=temp+self.mixture_coefficient[l]*math.exp(-0.5*np.dot(np.dot(np.transpose(self.data[i]-self.mui[l]),np.linalg.inv(self.cov[l])),self.data[i]-self.mui[l]))
                    Pm[i,j]=exp*self.mixture_coefficient[j]/temp

            for i in range(self.k):
                temp0=np.zeros(2)
                for s in range(self.data.shape[0]):
                    temp0=temp0+Pm[s,i]*self.data[s]
                self.mui[i]=temp0/np.sum(Pm[:,i])
                temp1=np.zeros((self.data.shape[1],self.data.shape[1]))

                for s in range(self.data.shape[0]):

                    temp1=temp1+Pm[s,i]*(self.data[s]-self.mui[i]).reshape(-1,1)*(self.data[s]-self.mui[i])


                self.cov[i]=temp1/np.sum(Pm[:,i])
                self.mixture_coefficient[i]=np.sum(Pm[:,i])/self.data.shape[0]



        clustering = np.zeros((self.data.shape[0]))

        for i in range(self.data.shape[0]):
            index = np.argmax(Pm[i])
            clustering[i] = index
        clustering = np.expand_dims(clustering, axis=1)
        self.data1 = np.concatenate((self.data, clustering), axis=1)
        self.data1 = self.data1[np.lexsort(self.data1.T)]



















test = np.loadtxt('b.txt')
case=Mixture_of_Gaussian(mixture_coefficent_initial=1/3,data=test)
print(case.cov)
case.update()
print(case.cov)
print(case.mui)
plt.scatter(case.data1[:,0],case.data1[:,1],c=case.data1[:,2],cmap="viridis")
plt.scatter(case.mui[:,0],case.mui[:,1],marker='*')
plt.show()