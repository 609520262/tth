import numpy as np
import matplotlib.pyplot as plt
import time
class K_armed_bandit():
    def __init__(self, K=3,eps=0.1,T=1000,P=[0.4,0.2,0.3]):

        self.K=K
        self.T=T
        self.P=np.array(P)
        self.eps=eps
    def update(self):
        plt.ion()  # 开启interactive mode 成功的关键函数
        plt.figure(1)
        y=[]
        x=[]
        r=0
        Q=np.zeros(self.K)
        count=np.zeros(self.K)
        for i in range(self.T):
            if np.random.rand()<self.eps:
                k=np.random.choice(self.K)
            else:
                k=self.P.argmax()
            print(r)
            v=np.random.choice([1,0],p=[self.P[k],1-self.P[k]])
            r=r+v
            x.append(i)
            y.append(r/i+1)
            Q[k]=(Q[k]*count[k]+v)/(count[k]+1)
            plt.clf()  # 清除之前画的图
            plt.plot(x, y, '-r')
            plt.pause(0.01)
            plt.ioff()








case=K_armed_bandit()
case.update()
