import random
import matplotlib.pyplot as plt
class KMeans():
    def __init__(self, k=3, max_iter=300,data=None):
        self.k=3
        self.data=data
        sample_list = [i for i in range(data.shape[0])]
        self.sample_num=random.sample(sample_list,k)
        self.initial=data[self.sample_num,:]
    def dist(self,mean_new):
        dist_array=np.zeros((self.data.shape[0],mean_new.shape[0]))
        clustering=np.zeros((self.data.shape[0]))
        n=self.data.shape[1]
        for i in range(self.data.shape[0]):
            for j in range(mean_new.shape[0]):
                dist_array[i][j]= np.sqrt(np.sum(np.square(self.data[i] -mean_new[j])))/n
            index = np.argmin(dist_array[i])
            clustering[i]=index

        #print(clustering)
        clustering=np.expand_dims(clustering,axis=1)
        self.data1=np.concatenate((self.data,clustering),axis=1)
        self.data1=self.data1[np.lexsort(self.data1.T)]

    def update(self,iter=None):
        self.iter=iter
        mean_new = np.zeros((self.k, self.data.shape[1]))
        self.dist(self.initial)
        for n in range(iter):
            print(self.data1)
            for i in range(self.k):
                num = np.where(self.data1[:, -1] == i)
                logit = self.data1[num]
                logit = np.squeeze(logit)
                # print(logit.shape)
                # print(np.mean(logit[:,:-1],axis=0))
                if n==iter-1:
                   plt.scatter(logit[:, 0], logit[:, 1])

                mean_new[i, :] = np.mean(logit[:, :-1], axis=0)

            self.mean_new = mean_new
            self.dist(mean_new)
        return self.mean_new








import numpy as np
test=np.loadtxt('b.txt')
print(test.shape)
plt.subplot(221)
case=KMeans(data=test)
plt.scatter(case.data[:,0],case.data[:,1])
plt.subplot(222)
print(case.update(iter=1))
plt.subplot(223)
print(case.update(iter=2))
plt.subplot(224)
print(case.update(iter=10))
plt.show()


