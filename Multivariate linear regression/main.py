import pandas as pd
import numpy as np
#定义多元线性回归模型
#输入样本，包含最后一列的y值
class Multivarite_linear_regression():
    def __init__(self, data=None):

        self.data=data
        self.data1=np.append(self.data[:,:-1],np.ones((self.data.shape[0], 1)),axis=1 )
        self.W=np.zeros((self.data.shape[1],1))
#采用最小二乘法得到w，b的闭式解
    def updata(self):
        A=np.dot(np.transpose(self.data1),self.data1)
        A1=np.linalg.inv(A)
        B=np.dot(A1,np.transpose(self.data1))
        C=self.data[:,-1].reshape(self.data.shape[0],1)
        self.W=np.dot(B,C)
        print('线性回归系数w：{}'.format(self.W[:-1,0]))
        print('线性回归系数b：{}'.format(self.W[-1, 0]))
#预测函数
    #输入：样本x值
    #输出：预测样本标签
    def predict(self,x_predict):
        print(x_predict)
        return np.dot(x_predict,self.W)






case=Multivarite_linear_regression(data=data)
print(case.W.shape)
case.updata()
x_predict=np.array([[0.00632,18,2.31,0,0.538,6.575,65.2,4.09,1,296,15.3,396.9,4.98,1]])
y_predict=case.predict(x_predict)
print(y_predict)












