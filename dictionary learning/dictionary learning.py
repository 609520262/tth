import pandas as pd
import numpy as np
from sklearn import linear_model

class dictionary():
    def __init__(self, data=None):
        self.data=data[:,:-1]
        U,S,V=np.linalg.svd(self.data)
        self.B=U
        print('字典矩阵的Size：{}'.format(self.B.shape))
        self.A=linear_model.orthogonal_mp(self.B, self.data)
        print('稀疏矩阵的Size：{}'.format(self.A.shape))

    def update(self,iter=10):
        for iter in range(iter):
            for i in range(self.B.shape[1]):
                X = self.data
                index = np.nonzero(X[i])[0]

                if len(index) == 0:
                    continue
                E = (X - np.dot(self.B, self.A))[:, index]
                U, S, V = np.linalg.svd(E)
                self.B[:, i] = U[:, 0]
            self.A = linear_model.orthogonal_mp(self.B, self.data)
















data = pd.read_csv("boston_housing_data.csv")
#print(data.shape)
data=data.dropna(axis=0,how='any',subset=['MEDV'])
print(data.shape)
data = np.array(data)
case=dictionary(data=data)
case.update()
print(case.B)


