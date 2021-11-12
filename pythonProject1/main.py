#创建特定数据的DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings
df_1=pd.DataFrame({'A' : 1.,
                    'B' : pd.Timestamp('20180310'),
                    'C' : pd.Series(1,index=['小红','小明','小白','小李'],dtype='float32'),
                    'D' : np.array([3] * 4,dtype='int32'),
                    'E' : ['test','train',np.nan,'train'],
                    'F' : 'foo'
                    },index=['小红','小明','小白','小李'],columns=['A','C','E','F','D'])
a=3
print("你好",a)
print(df_1)

print(df_1.index)
print(df_1['C'])
print(df_1[2:3])
print(df_1.loc[['小明','小李']])
print(df_1.iloc[3,3])
print(df_1.dropna(axis=0,how='any'))
print(len(df_1))
missing = df_1.isnull().sum()/len(df_1)
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()
print(df_1['A'].value_counts())#离散型变量
# g = sns.FacetGrid(df_1, col="A")
#g = g.map(plt.scatter, "A", "C", color="c")
#plt.show()
x, y = np.split(df_1, (1,), axis=1)
print(x)
print(y)