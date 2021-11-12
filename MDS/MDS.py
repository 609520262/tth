import pandas as pd
import numpy as np
class MDS():
    def __init__(self, data=None,k=3):
        self.data=data[:,:-1]
        #print(self.data.shape)
        dist=np.zeros((self.data.shape[0],self.data.shape[0]))
        self.B=np.zeros((self.data.shape[0],self.data.shape[0]))
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                dist[i,j]=np.dot(self.data[i,:],self.data[j,:].T)
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                dist_i=np.sum(dist[i,:])/dist.shape[0]
                dist_j=np.sum(dist[:,j])/dist.shape[0]
                dist__=np.sum(dist)/dist.shape[0]**2
                self.B[i,j]=-0.5*(dist[i,j]-dist_i-dist_j+dist__)
        #print(self.B.shape)
        value,eig=np.linalg.eig(self.B)
        sorted_indices=np.argsort(value)
        self.value=value[sorted_indices[:-k-1:-1]]
        self.eig=eig[:,sorted_indices[:-k-1:-1]]
    def update(self):
        return np.dot(self.eig, np.sqrt(self.value))



























data = pd.read_csv("boston_housing_data.csv")
#print(data.shape)
data=data.dropna(axis=0,how='any',subset=['MEDV'])
print(data.shape)
data = np.array(data)

case=MDS(data,k=5)
Z=case.update()
print(Z)

