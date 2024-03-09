
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
import pandas as pd
import shap

class Model1(nn.Module):
    def __init__(self,n1,n_feature):
        super(Model1,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n1, 16),
            nn.Sigmoid(),
            nn.Linear(16, n_feature),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.net(x)
        return x
n_feature = 4
f = 11
net1 = Model1(f,n_feature)

data_mimic = pd.read_csv(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\data\xg2.csv')
data_eicu = pd.read_csv(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\data\sepsis3_eicu_kmeans_drop_p2.csv')

feature = ["Mean PO2","Maximum HR","Mean HR","Mean MCHC","Mean RR","Maximum FiO2","Mean RDW","Maximum DBP","Mean BUN","Maximum Hemoglobin","Maximum SBP","Mean Albumin","Mean RBC","Maximum RR","Minimum RR","Minimum PO2","Maximum Albumin","Maximum Base Excess","Mean AG","Mean MCH"]
net1.load_state_dict(torch.load(r"E:\MIMIC\CCI\result\标准化\论文浓缩版\result\model\model_f_11.pth")['model_1'])
net1 = net1.to("cuda")
# for i in range(4):
x_train = data_mimic[feature[:f]].values
x_val = data_eicu[feature[:f]].values
x_train = (x_train - x_train.mean())/x_train.std()
x_val = (x_val - x_val.mean())/x_val.std()

x_train = torch.from_numpy(np.array(x_train, dtype="float32")).reshape((-1,f)).to("cuda")


x_val = torch.from_numpy(np.array(x_val, dtype="float32")).reshape((-1,f)).to("cuda")


n1 = 1000 #抽样1000个
n2 = 100 #重复100次
feature_important = np.zeros((4,n2,f))
Shap_values = np.zeros((4,n2*n1,f))
XX = np.zeros((4,n2*n1,f))
import random
for j in range(n2):
    print("第次",j)
    shap.initjs()
    selected_numbers = random.sample(list(range(len(x_train[:,0]))), n1)
    selected_numbers_val = random.sample(list(range(len(x_val[:,0]))), n1)
    explainer = shap.DeepExplainer(net1,x_train[selected_numbers])

    shap_values = explainer.shap_values(x_val[selected_numbers_val])
    print(shap_values[0].shape)
    for i in range(4):
        feature_important[i,j,:] = np.mean(np.abs(shap_values[i]), axis=0)
        Shap_values[i,j*n1:(j+1)*n1,:] = shap_values[i]
        XX[i,j*n1:(j+1)*n1,:] = x_val[selected_numbers_val].cpu().numpy()

for i in range(4):
    pd.DataFrame(feature_important[i]).to_csv(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\model\shap_f{}.csv'.format(i), index=False)

import seaborn as sns
sns.set(font='SimHei',font_scale = 0.6) #解决中文问题
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来显示负号
#
Title = ['Phenotype A','Phenotype B','Phenotype C','Phenotype D']
for i in range(4):
    xx = pd.DataFrame(XX[i,:,:].reshape((-1,f)))
    xx.columns = feature[:f]
    shap.summary_plot(np.array(Shap_values[i,:,:]).reshape((-1,f)), xx ,show=False)
    plt.xlabel('SHAP value of '+Title[i], fontsize=13)
    plt.savefig(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\model\shap_f1_{}.tif'.format(Title[i]), dpi=800)
    plt.show()

    # feature_important = np.sum(np.abs(shap_values[i]), axis=0)
    shap.summary_plot(np.array(Shap_values[i,:,:]).reshape((-1,f)), xx, plot_type="bar",show=False)
    plt.xlabel('mean(|SHAP value|) of '+Title[i], fontsize=13)
    plt.savefig(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\model\shap_f2_{}.tif'.format(Title[i]), dpi=800)
    plt.show()




class Model1(nn.Module):
    def __init__(self,n1,n_feature):
        super(Model1,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n1, 16),
            nn.Sigmoid(),
            nn.Linear(16, 32),
            nn.Sigmoid(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.net1 = nn.Sequential(
            nn.Linear(n1, 16),
            nn.Sigmoid(),
            nn.Linear(16, 32),
            nn.Sigmoid(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.net2 = nn.Sequential(
            nn.Linear(n1, 16),
            nn.Sigmoid(),
            nn.Linear(16, 32),
            nn.Sigmoid(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x1 = self.net(x)
        x2 = self.net1(x)
        x3 = self.net2(x)
        # x3 = x3 * (1-x2)
        x = torch.cat((x1,x2,x3),dim=-1)
        # x[:,0] = x[:,0]*x[:,1]
        # print(x.shape)
        return x
n_feature = 3
net1 = Model1(61,n_feature)
# net2 = Model2(38,n_feature)

data_mimic = pd.read_csv(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\data\xg2.csv')
data_eicu = pd.read_csv(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\data\sepsis3_eicu_kmeans_drop_p2.csv')
f = 61
feature = ["Mean HR","Minimum HR","Maximum HR","Mean SBP","Minimum SBP","Maximum SBP","Mean DBP","Minimum DBP","Maximum DBP","Mean MAP","Minimum MAP","Maximum MAP","Mean RR",
           "Minimum RR","Maximum RR","Mean Temperature","Minimum Temperature","Maximum Temperature","Mean PO2","Minimum PO2","Mean PCO2","Minimum PCO2","Maximum PCO2","Mean FiO2",
           "Minimum FiO2","Maximum FiO2","Mean Base Excess","Minimum Base Excess","Maximum Base Excess","Mean RBC","Maximum Hemoglobin","Mean RDW","Mean MCH","Mean MCV","Mean MCHC",
           "Mean Platelet","Mean WBC","Mean Basophils","Minimum Basophils","Maximum Basophils","Mean Eosinophils","Minimum Eosinophils","Mean Lymphocytes","Mean Monocytes",
           "Maximum Monocytes","Minimum Neutrophils","Mean Albumin","Minimum Albumin","Maximum Albumin","Mean AG","Mean BUN","Mean Calcium","Mean Chloride","Mean Sodium",
           "Mean Potassium","Mean Glucose","Minimum Glucose","Mean Creatinine","Mean INR","Mean PTT","Minimum PTT"]
x_train = data_mimic[feature[:f]].values

y_train = data_mimic[["hospital_expire_flag","acute_sepsis","cci"]].values
x_val = data_eicu[feature[:f]].values
y_val = data_eicu[["hosp_mort","acute_sepsis","cci"]].values
x_train = (x_train - x_train.mean())/x_train.std()
x_val = (x_val - x_val.mean())/x_val.std()

n_feature = 3
x_train = torch.from_numpy(np.array(x_train, dtype="float32")).reshape((-1,f)).to("cuda")
y_train = torch.from_numpy(np.array(y_train, dtype="float32")).reshape((-1,n_feature))
y_train = torch.tensor(y_train,dtype=torch.float32).to("cuda")
x_val = torch.from_numpy(np.array(x_val, dtype="float32")).reshape((-1,f)).to("cuda")
y_val = torch.from_numpy(np.array(y_val, dtype="float32")).reshape((-1,n_feature))
y_val = torch.tensor(y_val,dtype=torch.float32).to("cuda")

net1.load_state_dict(torch.load(r"E:\MIMIC\CCI\result\标准化\论文浓缩版\result\model\model_label_all.pth")['model_1'])

net1 = net1.to("cuda")
n1 = 1000 #抽样1000个
n2 = 100 #重复100次
feature_important = np.zeros((3,n2,61))
Shap_values = np.zeros((3,n2*n1,61))
XX = np.zeros((3,n2*n1,61))
for j in range(n2):
    print("第次",j)
    shap.initjs()
    import random
    selected_numbers = random.sample(list(range(len(x_train[:,0]))), n1)
    selected_numbers_val = random.sample(list(range(len(x_val[:,0]))), n1)
    explainer = shap.DeepExplainer(net1,x_train[selected_numbers])
    shap_values = explainer.shap_values(x_val[selected_numbers_val])
    print(shap_values[0].shape)
    for i in range(3):
        feature_important[i,j,:] = np.mean(np.abs(shap_values[i]), axis=0)
        Shap_values[i,j*n1:(j+1)*n1,:] = shap_values[i]
        XX[i,j*n1:(j+1)*n1,:] = x_val[selected_numbers_val].cpu().numpy()


for i in range(3):
    pd.DataFrame(feature_important[i]).to_csv(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\model\shap_all{}.csv'.format(i), index=False)

import seaborn as sns
sns.set(font='SimHei',font_scale = 0.6) #解决中文问题
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来显示负号
#
Title = ['Hospital Mortality','Acute Death','CCI']
for i in range(3):
    xx = pd.DataFrame(XX[i,:,:].reshape((-1,61)))
    xx.columns = feature
    shap.summary_plot(np.array(Shap_values[i,:,:]).reshape((-1,61)), xx ,show=False)
    plt.xlabel('SHAP value of '+Title[i], fontsize=13)
    plt.savefig(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\model\shap_all1_{}.tif'.format(i), dpi=800)
    plt.show()

    shap.summary_plot(np.array(Shap_values[i,:,:]).reshape((-1,61)), xx, plot_type="bar",show=False)
    plt.xlabel('mean(|SHAP value|) of '+Title[i], fontsize=13)
    plt.savefig(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\model\shap_all2_{}.tif'.format(i), dpi=800)
    plt.show()
