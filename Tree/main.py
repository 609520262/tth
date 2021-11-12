
import numpy as np
import pandas as pd
#定义决策树类
#目前该类包含的方法：1.计算节点信息增益 2.生成最大增益索引 3.节点划分 4.节点分割 5.生成树
#未来将会创建绘图方法将决策树可视化
#————————————————————————————————————————————————————————————————————————————————————————————————

#输入：完整数据集，带有标签
#输出：每一层的节点信息

#————————————————————————————————————————————————————————————————————————————————————————————————
#注意：这里采用的为西瓜数据集，读者可以根据需要自己调整，确保数据最后一列为标签
#查看决策树属性可根据Tree_layer属性获取


class Tree():
    def __init__(self,data=None):
        self.data=data
        self.Tree_layer=[]
        self.NodesDone=[]



        self.samples_num=self.data.shape[0]
#计算节点信息增益
    #输入：父级节点
    #输出：该节点的信息增益
    def Information_Entropy_Grain(self,Father_subset=None,Tree_node=None):
        Father_information=Father_subset.iloc[:,-1].unique()
        # 计算父级信息熵
        Ent_Father=0
        for temp in Father_information:
            Father_positive_ornot=Father_subset[Father_subset.iloc[:,-1].isin([temp])]
            pk_Father = Father_positive_ornot.shape[0] / Father_subset.shape[0]

            Ent_Father = Ent_Father + pk_Father * np.log2(pk_Father)
        Ent_Father=- Ent_Father

        # 计算子集信息熵
        class_information=Father_subset[Tree_node].unique()
        sum_Ent=0
        for temp_information in class_information:
            subset=Father_subset[Father_subset[Tree_node].isin([temp_information])]

            weight=subset.shape[0]/Father_subset.shape[0]

            positive_or_native_information = subset.iloc[:,-1].unique()
            Ent=0
            for temp in positive_or_native_information:
                subset_positive_ornot=subset[subset.iloc[:,-1].isin([temp])]
                pk=subset_positive_ornot.shape[0]/subset.shape[0]
                Ent=Ent+pk*np.log2(pk)

            sum_Ent=sum_Ent+weight*Ent

        #计算信息增益
        Gain= Ent_Father+sum_Ent
        return  Gain
        # print('属性\'{}\'的信息增益为{}'.format(Tree_node,Gain))
#生成最优节点索引
    #输入：父级集合
    #输出：子节点最优节点
    def Generate_maxGrain_dex(self,Father_subset=None):
        list_property_old = Father_subset.columns.values[:-1]
        list_property=[list for list in list_property_old if list not in self.NodesDone]

        list_Gain = np.array(
            [self.Information_Entropy_Grain(Father_subset=Father_subset, Tree_node=list) for list in list_property ])
        list_Gain = list_Gain.reshape(1, -1)
        # print(' list_Gain is{}:'.format( list_Gain))

        DataFrame_Gain = pd.DataFrame(list_Gain, index=['Grain'], columns=list_property)
        print(DataFrame_Gain)
        # 找出最优增益属性
        maxGrain_index = DataFrame_Gain.stack().idxmax(axis=1)[1]
        self.NodesDone.append( maxGrain_index)
        return maxGrain_index

        print(maxGrain_index)
#划分节点
    #输入：父级集合
    #输出：子层
    def spilt_TreeNodes(self,Father_subset=None,Tree_node=None):

        #根据最有增益属性划分集合

        maxGrain_index=self.Generate_maxGrain_dex(Father_subset=Father_subset)
        print('根据{}进行划分'.format( maxGrain_index))

        self.NodesDone.append(maxGrain_index)

        self.Tree_layer.append('{}'.format( maxGrain_index))
        self.Tree_layer.append('//')
        class_information = Father_subset[ maxGrain_index].unique()
        k=0
        d={}

        for temp_information in class_information:
            if len(Father_subset[Father_subset[maxGrain_index].isin([temp_information])].iloc[:,-1].unique())<2:
                if Father_subset[Father_subset[maxGrain_index].isin([temp_information])].iloc[:,-1].unique()[0]=='是':
                    self.Tree_layer.append('好瓜')
                else:
                    self.Tree_layer.append('坏瓜')
                continue
            elif len(Father_subset[maxGrain_index].unique())<2:
                 self.Tree_layer.append('好瓜或坏瓜')
                 continue
            else:
                d["{}{}".format(maxGrain_index,temp_information)] = Father_subset[Father_subset[maxGrain_index].isin([temp_information])]
                self.Tree_layer.append('{}=?'.format(temp_information))

                k = k + 1
        print(d)

        self.Tree_layer.append('//')
        # print(self.Tree_layer)
        return d

    def TreeGenerate(self):

        Son_subset= self.spilt_TreeNodes(Father_subset=case.data)
        print('************完成了根节点**************')
        i=1
        while Son_subset:

            list_keys = list(Son_subset.keys())

            list_Nodes={}
            for list_ket in list_keys:
                print(list_ket)


                GrandSon_subset=self.spilt_TreeNodes(Father_subset=Son_subset[list_ket])
                list_Nodes.update(GrandSon_subset)

            Son_subset=  list_Nodes
            print('-----------------------完成了{}层----------------'.format(i))
            i=i+1













































data=pd.read_csv('b.csv')#读取csv文件
data=data.iloc[:,1:]


case=Tree(data)
# case.Information_Entropy_Grain(Father_subset=case.data,Tree_node='纹理')
# case.spilt_TreeNodes(Father_subset=case.data)
case.TreeGenerate()
print(case.Tree_layer)

