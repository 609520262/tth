import numpy as np
import matplotlib.pyplot as plt
class SVM():
    def __init__(self, kernel='Gaussian', data=None,y=None,C=0.6):
        self.data=data
        self.label=y
        self.kernel=kernel
        self.K=np.zeros((self.data.shape[0],self.data.shape[0]))
        self.C=C
        sigma=1
        if kernel=='linear':
            for i in range(self.data.shape[0]):
                for j in range(self.data.shape[0]):
                   self.K[i,j]=np.sum(self.data[i,:]*self.data[j,:])
        elif kernel=='Gaussian':
            for i in range(self.data.shape[0]):
                for j in range(self.data.shape[0]):
                    self.K[i][j] = np.exp(-np.linalg.norm(self.data[i,:]-self.data[j,:])/(2* sigma^2))

        self.alpha=np.zeros(self.data.shape[0]).astype(np.float)
        self.b=0
        #print("核函数为{}".format(self.K))
    def Forward(self,i):
        return np.sum(self.alpha*self.label*self.K[i,:])+self.b

    def Heusistically_Search_1(self):

        for i in range(self.data.shape[0]):
            if self.alpha[i]<0 or self.alpha[i]>self.C :

                return i,self.Forward(i)-self.label[i]

                break
        if i==self.data.shape[0]-1:
            for i in range(self.data.shape[0]):
                if self.alpha[i] == 0 or self.alpha[i] == self.C:
                    return i, self.Forward(i) - self.label[i]

                    break


    def Heusistically_Search(self):
        [j,E1]=self.Heusistically_Search_1()
        disp=[]
        for j in range(self.data.shape[0]):
            E2=self.Forward(j)-self.label[j]
            disp.append(abs(E1-E2))
        print("E1-E2差值为{}，应选择第{}个alpha_i,对应E1{}".format(disp,disp.index(max(disp)),E2))
        return disp.index(max(disp)),E2
    def Clip(self,i,j):
        if self.label[i]!=self.label[j]:
            L=max(0,self.alpha[j]-self.alpha[i])
            H=min(self.C,self.C+self.alpha[j]-self.alpha[i])
            print("alphaj和alphai：{}，{}".format(self.alpha[j],self.alpha[i]))
            print("两个标签不一样L:{},H:{}".format(L, H))
        else:
            L=max(0,self.alpha[j]+self.alpha[i]-self.C)
            H=min(self.C,self.alpha[j]+self.alpha[i])
            print("两个标签一样L:{},H:{}".format(L, H))
        if self.alpha[j]>H:
            return H
        elif self.alpha[j]>=L and self.alpha[j]<=H:
            return self.alpha[j]
        else:
            return L


    def update(self,enpoch=20,tolerance=0.1):
        self.tolerance=tolerance
        for iter in range(enpoch):

            [i,E1]=self.Heusistically_Search_1()
            print("外循环选择了第{}个alphai,对应E1{}".format(i,E1))
            [j, E2] = self.Heusistically_Search()
            #print("alpha_i：{}，alpha_j:{}".format(i,j))
            yelta = self.K[i, i] + self.K[j, j] - 2 * self.K[i, j]
            print("yelta:{}".format(yelta))
            alphaj = self.alpha[j]
            self.alpha[j] = self.alpha[j] + self.label[j] * (E1 - E2) / yelta
            print("修减前alpha_j={}".format(self.alpha[i]))

            alphai=self.alpha[i]

            self.alpha[j] = self.Clip(i, j)
            print("修剪后alpha_j={}".format(self.alpha[j]))
            self.alpha[i] = self.alpha[i] + self.label[i] * self.label[j] * (alphaj - self.alpha[j] )
            print("更新alpha_i={}".format(self.alpha[i]))

            print("第{}轮的alpha为{}".format(iter, self.alpha))


            b1=self.b-(E1+self.label[i]*self.K[i,i]*(self.alpha[i]-alphai)+self.label[j]*self.K[i,j]*(self.alpha[j]-alphaj))
            b2=self.b-(E2+self.label[j]*self.K[i,j]*(self.alpha[i]-alphai)+self.label[j]*self.K[j,j]*(self.alpha[j]-alphaj))
            if self.alpha[i]>0 and self.alpha[i]<self.C:
                self.b=b1
            elif self.alpha[j]>0 and self.alpha[j]<self.C:
                self.b =b2
            else:
                 self.b=(b1+b2)/2
            print("b为{}".format(self.b))
                #Sum_S=0
        #S=np.where(self.alpha>0)[0]
        #for i in range(S.shape[0]):
           # t=S[i]
            #Sum_S=Sum_S+(1/self.label[t])-np.sum(self.alpha.take(S)*self.label.take(S)*self.K[:,t].take([S]))
        #self.b=Sum_S/S.shape[0]
    def plot_line(self,x):
        w=np.zeros((1,2))
        for i in range(self.data.shape[0]):
            w=self.alpha[i] * self.label[i] * self.data[i]+w

        for i in range(self.data.shape[0]):
            print(np.sum(self.data[i, :] * w) + self.b)


        a=w[0,0]
        m=w[0,1]
        y=-a*x/m-self.b/m
        print(m)
        plt.plot(x,y)









































data=np.loadtxt('3a.txt')
x=data[:,1:-1]
y=data[:,-1]

case=SVM(kernel='linear', data=x,y=y,C=0.6)

case.update()

plt.scatter(x[:9,0],x[:9,1])
plt.scatter(x[9:,0],x[9:,1])
case.plot_line(np.arange(1,2.0,0.1))

plt.show()


