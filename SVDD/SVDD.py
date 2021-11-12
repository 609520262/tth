import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

class SVDD():
    def __init__(self, kernel='Gaussian', data=None,y=None,C=1):
        self.data=data
        self.label=y
        self.kernel=kernel
        self.K=np.zeros((self.data.shape[0],self.data.shape[0]))
        self.C=C
        sigma=2
        if kernel=='linear':
            for i in range(self.data.shape[0]):
                for j in range(self.data.shape[0]):
                   self.K[i,j]=np.sum(self.data[i,:]*self.data[j,:])
        elif kernel=='Gaussian':
            for i in range(self.data.shape[0]):
                for j in range(self.data.shape[0]):
                    self.K[i,j] = np.exp(-np.sum(self.data[i,:]-self.data[j,:])**2/(2* sigma^2))

        self.alpha=np.random.uniform(0,self.C,self.data.shape[0]).astype(np.float)
        self.b=0
        self.a=np.zeros((1,self.data.shape[1]))
        #print("初始alpha{}".format(self.alpha))
    def Forward(self,i):
        return np.sum(self.alpha*self.K[:,i])
    def support_vector(self):
        sv=[i for i in range(self.alpha.shape[0]) if self.alpha[i]>0 and self.alpha[i]<self.C]
        return sv




    def Heusistically_Search(self):

        s_pre=[i for i in range(self.data.shape[0]) if self.alpha[i]<self.C]
        s_premax=[np.sum(self.a-self.data[i]**2) for i in s_pre]
        #print("s的候选属性{}".format(s_premax))
        s=s_pre[np.argmax(s_premax)]
        t_pre = [i for i in range(self.data.shape[0]) if self.alpha[i] >0 ]
        t_premin = [np.sum(self.a - self.data[i]**2) for i in t_pre]
        #print("t的候选属性{}".format(np.argmin(t_premin)))
        t = t_pre[np.argmin(t_premin)]
        return s,t,self.alpha[s]+self.alpha[t]

    def Clip(self,s,t,gamma):
        if self.alpha[s]<0:
            self.alpha[s]=0
            self.alpha[t]=gamma
        elif self.alpha[s]<0:
            self.alpha[t] = 0
            self.alpha[s] = gamma
        elif self.alpha[s] >self.C:
            self.alpha[s]=self.C
            self.alpha[t]=gamma-self.C
        elif self.alpha[t] > self.C:
            self.alpha[t] = self.C
            self.alpha[s] = gamma - self.C
    def acc(self):
        sums=0
        errs=0
        for i, alpha in enumerate(self.alpha):
            sums = alpha * self.label[i] * np.sum(self.label * self.alpha * self.K[i, :]) + sums
        for i,alpha in enumerate(self.data):
            d=np.sqrt(self.K[i,i]-2*np.sum(self.label*self.alpha*self.K[:,i])+ sums)
            if d<=np.mean(self.r) and self.label[i]==-1 or d>=np.mean(self.r) and self.label[i]==1:
                errs=errs+1
        return 1-errs/self.data.shape[0]



    def update(self,enpoch=20,tolerance=0.1):
        self.tolerance=tolerance
        for iter in range(enpoch):
            [s,t,gamma]=self.Heusistically_Search()
            yelta = 2 * self.K[s, t] - self.K[s, s] - self.K[t, t]
            self.alpha[t]=self.alpha[t]+(self.Forward(t)-self.Forward(s))/yelta
            self.alpha[s]=gamma-self.alpha[t]
            self.Clip(s,t,gamma)
            suma=0

            sv=self.support_vector()
            sums=0
            suma=np.zeros((1,self.data.shape[1]))

            for i,alpha in enumerate(self.alpha):
                sums=alpha*self.label[i]*np.sum(self.label*self.alpha*self.K[i,:])+sums
                suma=alpha*self.label[i]*self.data[i,:]+suma
            #print("支持向量为：{}".format(sv))
            self.a=suma
            self.r=[np.sqrt(self.K[i,i]-2*np.sum(self.label*self.alpha*self.K[:,i])+ sums) for i in sv]
            acc=self.acc()
            print("acc{}".format(acc))
            #print("圆心{}".format(self.a))

            #print("第{}轮alpha:{}".format(iter,self.alpha))




    def plot_circle(self):
        n=len(self.r)
        theta=np.arange(0,2*np.pi,0.01)
        x=np.zeros(theta.shape[0])
        y=np.zeros(theta.shape[0])

        for i in range(n):

            x[(i*np.round(theta.shape[0]/n)).astype(int):(i*np.round(theta.shape[0]/n)+np.round(theta.shape[0]/n)).astype(int)]=(self.a[0,0]+self.r[i]*np.cos(theta[(i*np.round(theta.shape[0]/n)).astype(int):(i*np.round(theta.shape[0]/n)+np.round(theta.shape[0]/n)).astype(int)]))

            y[(i*np.round(theta.shape[0]/n)).astype(int):(i*np.round(theta.shape[0]/n)+np.round(theta.shape[0]/n)).astype(int)]=(self.a[0,1]+self.r[i]*np.sin(theta[(i*np.round(theta.shape[0]/n)).astype(int):(i*np.round(theta.shape[0]/n)+np.round(theta.shape[0]/n)).astype(int)]))
        print(x.shape,y.shape)
        plt.plot(x,y)




x,y = datasets.make_moons(n_samples = 100, noise = 0.2, random_state = 42)
y = 2*y - 1 # Rescale labels to be {-1,1}

case=SVDD( kernel='linear', data=x,y=y,C=10)
case.update()
plt.scatter(x[:,0],x[:,1],c=y)

case.plot_circle()


plt.show()