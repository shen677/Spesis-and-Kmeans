import miceforest as mf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(font='SimHei',font_scale = 0.6) #解决中文问题
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来显示负号

data = pd.read_csv(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\data\sepsis3_cci.csv',encoding='utf-8') #mimic
# data = pd.read_csv(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\data\eicu_cci.csv',encoding='utf-8') #eicu

print(data.values.shape)
print(np.sum(data.isnull()))

# This part is specific to the MIMIC database, so comment it out if you are running interpolation for the eICU database.
drop_colnames = []
for i in range(len(data.columns)):
  null = np.sum(data.iloc[:,i].isnull())
  if null >0 and null <=5:
    data.iloc[data.iloc[:,i].isnull(), i] = data.iloc[:,i].mean()
  if null/len(data.index) >= 0.5:
    drop_colnames.append(data.columns[i])
for c in drop_colnames:
    data = data.drop(c, axis=1)
    print("删除",c)


# # 单一插补
data_miss0 = data.iloc[:,17:17+111-len(drop_colnames)].copy() # mimic
# data_miss0 = data.iloc[:,21:21+108].copy() #eicu

# 3-2多重插补 ：使用MultipleImputedKernel实例化多重插补对象
kernel = mf.MultipleImputedKernel(
  data_miss0,
  datasets = 5,
  save_all_iterations=True,
  save_models =1,
  random_state=1991
)

# Run the MICE algorithm for 3 iterations on each of the datasets
kernel.mice(iterations= 3,
            n_jobs=-1
           )

print(kernel)
# # 插补值的分布

kernel.plot_imputed_distributions(left=0.1, right=0.9, bottom=0.1, top=0.95, wspace=0.6, hspace=0.6)

plt.savefig(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\mice\mice_mimic.tif', dpi=600)
# plt.savefig(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\mice\mice_eicu.tif', dpi=600)
plt.show()


#分析并使用多重填补结果
dataresult=[]
result=[]
result1=[]
for i in range(kernel.dataset_count()):
  dataresult.append(kernel.complete_data(i))
  dd=(((dataresult[i].mean()-data_miss0.mean()))/data_miss0.mean()*100)
  # dd.to_csv('result' + str(i) + '.csv')
  result.append(dd)
  result1.append(dd.sum())
print(result[0].shape)

#可以取每个数据集离原来均值最小的
name=data_miss0.columns
new_complete=pd.DataFrame(columns=name)
lst=[]#储存要哪个数据集
for i in range(len(name)):
    re = []
    for j in range(kernel.dataset_count()):
        print(j,i)
        re.append(result[j][i])
    a=re.index(min(re))   #返回最小值所在数据集，返回在数据集中是第几个数据集有最符合插补值
    lst.append(a)
for i in range(len(name)):
    new_complete[name[i]]=dataresult[lst[i]][name[i]]

data.iloc[:,17:17+108] = new_complete # mimic
# data.iloc[:,21:21+108] = new_complete #eicu

# data.to_csv(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\data\sepsis3_cci_mice_drop.csv', index=False) # mimic
# data.to_csv(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\data\sepsis3_cci_eicu_drop2.csv', index=False) # eicu

print(new_complete)
print(result1)

