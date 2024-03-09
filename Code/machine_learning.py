import warnings
warnings.filterwarnings("ignore")
import torch
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import xgboost as xgb
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import scipy
sns.set(font='SimHei',font_scale = 0.8) #解决中文问题
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来显示负号

def AUC(y, out1, pplot=False):
    dict1 = {
        "Precision": [],
        "Recall": [],
        "F1": [],
        "specificity": [],
        "acc": [],
        "roc_auc": [],
        "p": []
    }

    for i in range(4):

        OUT1 = out1
        OUT2 = OUT1.copy()
        OUT2[:, i] = -100
        OUT = OUT1[:, i] - OUT2.max(axis=-1)
        y1 = np.zeros(y.shape)
        y1[y == i] = 1
        Y = y1.reshape((-1))
        fpr_BPNN, tpr_BPNN, thresholds_BPNN = roc_curve(Y, OUT)
        # else:
        #     y = y1[out1<0.5,1]
        #     y = y.long()
        #     Y = y.detach().numpy()
        #     OUT = out2[out1<0.5].detach().numpy()
        #     fpr_BPNN, tpr_BPNN, thresholds_BPNN = roc_curve(Y, OUT)
        dict1["roc_auc"].append(auc(fpr_BPNN, tpr_BPNN))
        stats, p1 = scipy.stats.ranksums(OUT[Y == 1],
                                         OUT[Y == 0], alternative="greater")
        dict1["p"].append(p1)
        C = confusion_matrix(Y, np.round(OUT))
        dict1["Precision"].append(C[1, 1] / (C[1, 1] + C[0, 1]))  # sensitivity
        dict1["Recall"].append(C[1, 1] / (C[1, 1] + C[1, 0]))
        dict1["specificity"].append(C[0, 0] / (C[0, 0] + C[1, 0]))
        dict1["acc"].append((C[0, 0] + C[1, 1]) / sum(sum(C)))
        dict1["F1"].append(
            2 * dict1["Precision"][-1] * dict1["Recall"][-1] / (dict1["Recall"][-1] + dict1["Precision"][-1]))
        if pplot:
            plt.figure()
            lw = 2
            plt.plot(fpr_BPNN, tpr_BPNN, color='red',
                     lw=lw, label='ROC (AUC_BPNN = %0.4f)' % dict1["roc_auc"][-1])
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    return dict1

def AUC1(y, out1, pplot=False):
    dict1 = {
        "Precision": [],
        "Recall": [],
        "F1": [],
        "specificity": [],
        "acc": [],
        "roc_auc": [],
        "p": []
    }

    for i in range(1):

        OUT = out1
        Y = y
        fpr_BPNN, tpr_BPNN, thresholds_BPNN = roc_curve(Y, OUT)
        # else:
        #     y = y1[out1<0.5,1]
        #     y = y.long()
        #     Y = y.detach().numpy()
        #     OUT = out2[out1<0.5].detach().numpy()
        #     fpr_BPNN, tpr_BPNN, thresholds_BPNN = roc_curve(Y, OUT)
        dict1["roc_auc"].append(auc(fpr_BPNN, tpr_BPNN))
        stats, p1 = scipy.stats.ranksums(OUT[Y == 1],
                                         OUT[Y == 0], alternative="greater")
        dict1["p"].append(p1)
        C = confusion_matrix(Y, np.round(OUT))
        dict1["Precision"].append(C[1, 1] / (C[1, 1] + C[0, 1]))  # sensitivity
        dict1["Recall"].append(C[1, 1] / (C[1, 1] + C[1, 0]))
        dict1["specificity"].append(C[0, 0] / (C[0, 0] + C[1, 0]))
        dict1["acc"].append((C[0, 0] + C[1, 1]) / sum(sum(C)))
        dict1["F1"].append(
            2 * dict1["Precision"][-1] * dict1["Recall"][-1] / (dict1["Recall"][-1] + dict1["Precision"][-1]))
        if pplot:
            plt.figure()
            lw = 2
            plt.plot(fpr_BPNN, tpr_BPNN, color='red',
                     lw=lw, label='ROC (AUC_BPNN = %0.4f)' % dict1["roc_auc"][-1])
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    return dict1





# Development of a sepsis subphenotype prediction model by BPNN, XGBoost, SVM, RF
data_mimic = pd.read_csv(r"E:\MIMIC\CCI\result\标准化\论文浓缩版\data\xg2.csv",encoding='utf-8') # mimic
data_eicu = pd.read_csv(r"E:\MIMIC\CCI\result\标准化\论文浓缩版\data\sepsis3_eicu_kmeans_drop_p2.csv",encoding='utf-8') # eicu
Z = np.zeros((20,8))
for f in range(1,21):
    feature = ["Mean PO2","Maximum HR","Mean HR","Mean MCHC","Mean RR","Maximum FiO2","Mean RDW","Maximum DBP","Mean BUN","Maximum hemoglobin","Maximum SBP","Mean Albumin","Mean RBC","Maximum RR","Minimum HR","Minimum PO2","Maximum Albumin","Maximum Base Excess","Mean AG","Mean MCH"]
    x_train = data_mimic[feature[:f]].values
    y_train = data_mimic["kmeans"].values
    x_val = data_eicu[feature[:f]].values
    y_val = data_eicu["kmeans"].values
    x_train = (x_train - x_train.mean())/x_train.std()
    x_val = (x_val - x_val.mean())/x_val.std()

    x_train = torch.from_numpy(np.array(x_train, dtype="float32")).reshape((-1,f))
    y_train = torch.from_numpy(np.array(y_train, dtype="float32")).reshape((-1,1))
    y_train = torch.tensor(y_train,dtype=torch.long)
    x_val = torch.from_numpy(np.array(x_val, dtype="float32")).reshape((-1,f))
    y_val = torch.from_numpy(np.array(y_val, dtype="float32")).reshape((-1,1))
    y_val = torch.tensor(y_val,dtype=torch.long)



    from torch import nn
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
    net1 = Model1(f,n_feature)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(net1.parameters(), lr=0.03,betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, amsgrad=False) # 优化器：Adam
    def train(x,y,model1):
        model1.train()

        out1 = model1(x).reshape((-1,4))  # Perform a single forward pass.
        loss = criterion(out1, y.reshape(-1))# Compute the loss.
        optim.zero_grad()  # Clear gradients.
        loss.backward()  # Derive gradients.
        optim.step()  # Update parameters based on gradients.
        return AUC(y,out1)



    def test(x,y,model1,pplot=False):
        model1.eval()
        with torch.no_grad():
            out1 = model1(x).reshape((-1,4))

        return AUC(y,out1)  # Derive ratio of correct predictions.

    from sklearn.metrics import roc_curve, auc,confusion_matrix
    from matplotlib import pyplot as plt
    import seaborn as sns
    import scipy
    sns.set(font='SimHei',font_scale = 0.8) #解决中文问题
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False #用来显示负号
    def AUC(y,out1,pplot=False):
        dict1 = {
            "Precision":[],
            "Recall": [],
            "F1": [],
            "specificity": [],
            "acc": [],
            "roc_auc": [],
            "p": []
        }

        for i in range(4):
            OUT1 = out1.detach().numpy()
            OUT2 = OUT1.copy()
            OUT2[:,i] = -100
            OUT = OUT1[:,i] - OUT2.max(axis = -1)
            y1 = np.zeros(y.detach().numpy().shape)
            y1[y.detach().numpy()==i] = 1
            Y = y1.reshape((-1))
            fpr_BPNN, tpr_BPNN, thresholds_BPNN = roc_curve(Y, OUT)
            dict1["roc_auc"].append(auc(fpr_BPNN, tpr_BPNN))
            stats, p1 = scipy.stats.ranksums(OUT[Y == 1],
                                            OUT[Y == 0], alternative="greater")
            dict1["p"].append(p1)
            C = confusion_matrix(Y, np.round(OUT))
            dict1["Precision"].append(C[1, 1] / (C[1, 1] + C[0, 1]))  # sensitivity
            dict1["Recall"].append(C[1, 1] / (C[1, 1] + C[1, 0]))
            dict1["specificity"].append(C[0, 0] / (C[0, 0] + C[1, 0]))
            dict1["acc"].append((C[0, 0] + C[1, 1]) / sum(sum(C)))
            dict1["F1"].append(2 * dict1["Precision"][-1] * dict1["Recall"][-1] / (dict1["Recall"][-1] + dict1["Precision"][-1]))
            if pplot:
                plt.figure()
                lw = 2
                plt.plot(fpr_BPNN, tpr_BPNN, color='red',
                         lw=lw, label='ROC (AUC_BPNN = %0.4f)' % dict1["roc_auc"][-1])
                plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        return dict1


    from scipy.stats import norm
    def CI(auc, label, alpha = 0.05):
        label = np.array(label)#防止label不是array类型
        n1, n2 = np.sum(label == 1), np.sum(label == 0)
        q1 = auc / (2-auc)
        q2 = (2 * auc ** 2) / (1 + auc)
        se = np.sqrt(abs((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n2 -1) * (q2 - auc ** 2)) / (n1 * n2)))
        confidence_level = 1 - alpha
        z_lower, z_upper = norm.interval(confidence_level)
        se =z_upper * se
        return auc-se,auc+se

    auc_test_max = 0
    for epoch in range(400):
        if epoch % 100 == 0:
            print(epoch)
        L_train = train(x_train,y_train,net1)
        L_test = test(x_val,y_val,net1)
        if np.mean(L_test["roc_auc"]) >auc_test_max:
            auc_test_max = np.mean(L_test["roc_auc"])
            print(np.mean(L_train["roc_auc"]),"\n",np.mean(L_test["roc_auc"]))
            torch.save({'model_1': net1.state_dict()}, r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\model\model_f_{}.pth'.format(f))
            Z[f-1,:4] = np.array(L_train["roc_auc"])
            Z[f - 1, 4:] = np.array(L_test["roc_auc"])
    net1.load_state_dict(torch.load(r"E:\MIMIC\CCI\result\标准化\论文浓缩版\result\model\model_f_{}.pth".format(f))['model_1'])
    with torch.no_grad():
        L_train = test(x_train,y_train,net1)
        L_test = test(x_val,y_val,net1)
        Z[f - 1, :4] = np.array(L_train["roc_auc"])
        Z[f - 1, 4:] = np.array(L_test["roc_auc"])
print(Z)
pd.DataFrame(Z).to_csv(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\model\BPNN.csv', index=False)

data_mimic = pd.read_csv('data/xg2.csv')
data_eicu = pd.read_csv('data/sepsis3_eicu_kmeans_drop_p2.csv')
Z = np.zeros((20,8))
for f in range(1,21):
    feature = ["Mean PO2","Maximum HR","Mean HR","Mean MCHC","Mean RR","Maximum FiO2","Mean RDW","Maximum DBP","Mean BUN","Maximum hemoglobin","Maximum SBP","Mean Albumin","Mean RBC","Maximum RR","Minimum HR","Minimum PO2","Maximum Albumin","Maximum Base Excess","Mean AG","Mean MCH"]
    x_train = data_mimic[feature[:f]].values
    y_train = data_mimic["kmeans"].values
    x_val = data_eicu[feature[:f]].values
    y_val = data_eicu["kmeans"].values
    x_train = (x_train - x_train.mean())/x_train.std()
    x_val = (x_val - x_val.mean())/x_val.std()

    clf = xgb.XGBClassifier(
        n_estimators=30,  # 迭代次数
        learning_rate=0.08,  # 步长
        max_depth=5,  # 树的最大深度
        min_child_weight=1,  # 决定最小叶子节点样本权重和
        silent=1,  # 输出运行信息
        subsample=0.8,  # 每个决策树所用的子样本占总样本的比例（作用于样本）
        colsample_bytree=0.8,  # 建立树时对特征随机采样的比例（作用于特征）典型值：0.5-1
        objective='multi:softmax',  # 多分类！！！！！
        eta=0.8, #了防止过拟合，更新过程中用到的收缩步长。eta通过缩减特征 的权重使提升计算过程更加保守。缺省值为0.3，取值范围为：[0,1]
        num_class=4,
        nthread=4,
        seed=1991)
    print( "training...")
    clf.fit(x_train, y_train, verbose=True)
    fit_pred = clf.predict_proba(x_val)
    print(fit_pred.shape)
    L_test = AUC(y_val, fit_pred)
    L_train = AUC(y_train, clf.predict_proba(x_train))
    print(np.mean(L_train["roc_auc"]), "\n", np.mean(L_test["roc_auc"]))
    Z[f - 1, :4] = np.array(L_train["roc_auc"])
    Z[f - 1, 4:] = np.array(L_test["roc_auc"])
print(Z)
pd.DataFrame(Z).to_csv(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\model\XGOOST.csv', index=False)

from sklearn.svm import SVC
Z = np.zeros((20,8))
for f in range(1,21):
    feature = ["Mean PO2", "Maximum HR", "Mean HR", "Mean MCHC", "Mean RR", "Maximum FiO2", "Mean RDW", "Maximum DBP",
               "Mean BUN", "Maximum hemoglobin", "Maximum SBP", "Mean Albumin", "Mean RBC", "Maximum RR", "Minimum HR",
               "Minimum PO2", "Maximum Albumin", "Maximum Base Excess", "Mean AG", "Mean MCH"]
    x_train = data_mimic[feature[:f]].values
    y_train = data_mimic["kmeans"].values
    x_val = data_eicu[feature[:f]].values
    y_val = data_eicu["kmeans"].values
    x_train = (x_train - x_train.mean())/x_train.std()
    x_val = (x_val - x_val.mean())/x_val.std()

    clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
        max_iter=-1, probability=True, random_state=1991, shrinking=True,
        tol=0.001, verbose=False)
    print("training...")
    clf.fit(x_train, y_train)
    fit_pred = clf.predict_proba(x_val)
    print(fit_pred.shape)
    L_test = AUC(y_val, fit_pred)
    L_train = AUC(y_train, clf.predict_proba(x_train))
    print(np.mean(L_train["roc_auc"]), "\n", np.mean(L_test["roc_auc"]))
    Z[f - 1, :4] = np.array(L_train["roc_auc"])
    Z[f - 1, 4:] = np.array(L_test["roc_auc"])
print(Z)
pd.DataFrame(Z).to_csv(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\model\SVM.csv', index=False)

from sklearn.ensemble import RandomForestClassifier

Z = np.zeros((20,8))
for f in range(1,21):
    feature = ["Mean PO2", "Maximum HR", "Mean HR", "Mean MCHC", "Mean RR", "Maximum FiO2", "Mean RDW", "Maximum DBP",
               "Mean BUN", "Maximum hemoglobin", "Maximum SBP", "Mean Albumin", "Mean RBC", "Maximum RR", "Minimum HR",
               "Minimum PO2", "Maximum Albumin", "Maximum Base Excess", "Mean AG", "Mean MCH"]
    x_train = data_mimic[feature[:f]].values
    y_train = data_mimic["kmeans"].values
    x_val = data_eicu[feature[:f]].values
    y_val = data_eicu["kmeans"].values
    x_train = (x_train - x_train.mean())/x_train.std()
    x_val = (x_val - x_val.mean())/x_val.std()
    # 全部参数
    clf = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=5, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=30,
                           n_jobs=-1, oob_score=True, random_state=1991,
                           verbose=0, warm_start=False)
    print("training...")
    clf.fit(x_train, y_train)
    fit_pred = clf.predict_proba(x_val)
    print(fit_pred.shape)
    L_test = AUC(y_val, fit_pred)
    L_train = AUC(y_train, clf.predict_proba(x_train))
    print(np.mean(L_train["roc_auc"]), "\n", np.mean(L_test["roc_auc"]))
    Z[f - 1, :4] = np.array(L_train["roc_auc"])
    Z[f - 1, 4:] = np.array(L_test["roc_auc"])
print(Z)
pd.DataFrame(Z).to_csv(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\model\RF.csv', index=False)



# choose BPNN and feature numbers = 11 as sepsis subphenotype prediction model
from torch import nn
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

f = 11
feature = ["Mean PO2", "Maximum HR", "Mean HR", "Mean MCHC", "Mean RR", "Maximum FiO2", "Mean RDW", "Maximum DBP",
               "Mean BUN", "Maximum hemoglobin", "Maximum SBP", "Mean Albumin", "Mean RBC", "Maximum RR", "Minimum HR",
               "Minimum PO2", "Maximum Albumin", "Maximum Base Excess", "Mean AG", "Mean MCH"]
x_train = data_mimic[feature[:f]].values
x_val = data_eicu[feature[:f]].values
x_train = (x_train - x_train.mean())/x_train.std()
x_val = (x_val - x_val.mean())/x_val.std()
n_feature = 4
x_train = torch.from_numpy(np.array(x_train, dtype="float32")).reshape((-1,f))
x_val = torch.from_numpy(np.array(x_val, dtype="float32")).reshape((-1,f))
net1 = Model1(f, n_feature)
net1.load_state_dict(torch.load(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\model\model_f_11.pth')['model_1'])

net1.eval()
def aucc(out):
    OUT = np.zeros(out.shape)
    for i in range(n_feature):
        OUT1 = out
        OUT2 = OUT1.copy()
        OUT2[:, i] = -100
        OUT[:,i] = OUT1[:, i] - OUT2.max(axis=-1)
    return OUT
with torch.no_grad():
    out1 = net1(x_train).reshape((-1,n_feature)).detach().numpy()
    out1 = aucc(out1)
    pd.DataFrame(out1).to_csv(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\data\BPNN_subphenotype_mimic.csv', index=False)
    out2 = net1(x_val).reshape((-1,n_feature)).detach().numpy()
    out2 = aucc(out2)
    pd.DataFrame(out2).to_csv(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\data\BPNN_subphenotype_eicu.csv', index=False)


### Selection Primary Risk Factor Prediction Models for Adverse Outcomes
data_mimic = pd.read_csv(r"E:\MIMIC\CCI\result\标准化\论文浓缩版\data\xg2.csv",encoding='utf-8') # mimic
data_eicu = pd.read_csv(r"E:\MIMIC\CCI\result\标准化\论文浓缩版\data\sepsis3_eicu_kmeans_drop_p2.csv",encoding='utf-8') # eicu
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
x_train = torch.from_numpy(np.array(x_train, dtype="float32")).reshape((-1,f))
y_train = torch.from_numpy(np.array(y_train, dtype="float32")).reshape((-1,n_feature))
y_train = torch.tensor(y_train,dtype=torch.float32)
x_val = torch.from_numpy(np.array(x_val, dtype="float32")).reshape((-1,f))
y_val = torch.from_numpy(np.array(y_val, dtype="float32")).reshape((-1,n_feature))
y_val = torch.tensor(y_val,dtype=torch.float32)



from torch import nn
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
        x = torch.cat((x1,x2,x3),dim=-1)
        return x


def train(x,y,model1):
    model1.train()

    out1 = model1(x).reshape((-1,n_feature))  # Perform a single forward pass.
    loss = criterion(out1, y.reshape(-1,n_feature))# Compute the loss.
    optim.zero_grad()  # Clear gradients.
    loss.backward()  # Derive gradients.
    optim.step()  # Update parameters based on gradients.
    return AUC(y,out1)

def test(x,y,model1,pplot=False):
    model1.eval()
    with torch.no_grad():
        out1 = model1(x).reshape((-1,n_feature))

    return AUC(y,out1)  # Derive ratio of correct predictions.

from sklearn.metrics import roc_curve, auc,confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
import scipy
sns.set(font='SimHei',font_scale = 0.8) #解决中文问题
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来显示负号
def AUC(y,out1,pplot=False):
    dict1 = {
        "Precision":[],
        "Recall": [],
        "F1": [],
        "specificity": [],
        "acc": [],
        "roc_auc": [],
        "p": []
    }
    for i in range(n_feature):
        OUT = out1.detach().numpy()[:,i]
        Y = y.detach().numpy().reshape((-1,n_feature))[:,i]
        fpr_BPNN, tpr_BPNN, thresholds_BPNN = roc_curve(Y, OUT)
        dict1["roc_auc"].append(auc(fpr_BPNN, tpr_BPNN))
        stats, p1 = scipy.stats.ranksums(OUT[Y == 1],
                                        OUT[Y == 0], alternative="greater")
        dict1["p"].append(p1)
        C = confusion_matrix(Y, np.round(OUT))
        dict1["Precision"].append(C[1, 1] / (C[1, 1] + C[0, 1]))  # sensitivity
        dict1["Recall"].append(C[1, 1] / (C[1, 1] + C[1, 0]))
        dict1["specificity"].append(C[0, 0] / (C[0, 0] + C[1, 0]))
        dict1["acc"].append((C[0, 0] + C[1, 1]) / sum(sum(C)))
        dict1["F1"].append(2 * dict1["Precision"][-1] * dict1["Recall"][-1] / (dict1["Recall"][-1] + dict1["Precision"][-1]))
        if pplot:
            plt.figure()
            lw = 2
            plt.plot(fpr_BPNN, tpr_BPNN, color='red',
                     lw=lw, label='ROC (AUC_BPNN = %0.4f)' % dict1["roc_auc"][-1])
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    return dict1



out1 = np.zeros(y_train.shape)
out2 = np.zeros(y_val.shape)
net1 = Model1(f, n_feature)
criterion = nn.BCELoss()
optim = torch.optim.Adam(net1.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00001,
                         amsgrad=False)  # 优化器：Adam
auc_test_max = 0
for epoch in range(4000):
    L_train = train(x_train,y_train,net1)
    L_test = test(x_val,y_val,net1)
    if np.mean(L_test["roc_auc"]) >auc_test_max:
        auc_test_max = np.mean(L_test["roc_auc"])
        # print(np.mean(L_train["roc_auc"]),"\n",np.mean(L_test["roc_auc"]))
        print(L_train["roc_auc"], "\n", L_test["roc_auc"])
        torch.save({'model_1': net1.state_dict()}, r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\model\model_label_all.pth') # 61：model_label_all2.pth

net1.load_state_dict(torch.load(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\model\model_label_all.pth')['model_1'])
net1.eval()
# net2.eval()
with torch.no_grad():
    out1 = net1(x_train).reshape((-1,n_feature)).detach().numpy()
    out2 = net1(x_val).reshape((-1,n_feature)).detach().numpy()
    L_train = test(x_train, y_train, net1)
    L_test = test(x_val,y_val,net1)
    print(L_train["roc_auc"], "\n", L_test["roc_auc"])
pd.DataFrame(out1).to_csv(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\model\BPNN_mimic_all.csv', index=False)
pd.DataFrame(out2).to_csv(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\model\BPNN_eicu_all.csv', index=False)

f = 61
feature = ["Mean HR","Minimum HR","Maximum HR","Mean SBP","Minimum SBP","Maximum SBP","Mean DBP","Minimum DBP","Maximum DBP","Mean MAP","Minimum MAP","Maximum MAP","Mean RR",
           "Minimum RR","Maximum RR","Mean Temperature","Minimum Temperature","Maximum Temperature","Mean PO2","Minimum PO2","Mean PCO2","Minimum PCO2","Maximum PCO2","Mean FiO2",
           "Minimum FiO2","Maximum FiO2","Mean Base Excess","Minimum Base Excess","Maximum Base Excess","Mean RBC","Maximum Hemoglobin","Mean RDW","Mean MCH","Mean MCV","Mean MCHC",
           "Mean Platelet","Mean WBC","Mean Basophils","Minimum Basophils","Maximum Basophils","Mean Eosinophils","Minimum Eosinophils","Mean Lymphocytes","Mean Monocytes",
           "Maximum Monocytes","Minimum Neutrophils","Mean Albumin","Minimum Albumin","Maximum Albumin","Mean AG","Mean BUN","Mean Calcium","Mean Chloride","Mean Sodium",
           "Mean Potassium","Mean Glucose","Minimum Glucose","Mean Creatinine","Mean INR","Mean PTT","Minimum PTT"]
data_mimic = pd.read_csv(r"E:\MIMIC\CCI\result\标准化\论文浓缩版\data\xg2.csv",encoding='utf-8') # mimic
data_eicu = pd.read_csv(r"E:\MIMIC\CCI\result\标准化\论文浓缩版\data\sepsis3_eicu_kmeans_drop_p2.csv",encoding='utf-8') # eicu

x_train = data_mimic[feature[:f]].values
y_train = data_mimic[["hospital_expire_flag","acute_sepsis","cci"]].values
x_val = data_eicu[feature[:f]].values
y_val = data_eicu[["hosp_mort","acute_sepsis","cci"]].values
x_train = (x_train - x_train.mean())/x_train.std()
x_val = (x_val - x_val.mean())/x_val.std()
X_mimic = np.zeros((y_train.shape))
X_eicu = np.zeros((y_val.shape))

for i in range(3):
    clf = xgb.XGBClassifier(
        n_estimators=20,  # 迭代次数
        learning_rate=0.08,  # 步长
        max_depth=5,  # 树的最大深度
        min_child_weight=1,  # 决定最小叶子节点样本权重和
        silent=1,  # 输出运行信息
        subsample=0.8,  # 每个决策树所用的子样本占总样本的比例（作用于样本）
        colsample_bytree=0.8,  # 建立树时对特征随机采样的比例（作用于特征）典型值：0.5-1
        objective='binary:logistic',  # 多分类！！！！！
        eta=0.8, #了防止过拟合，更新过程中用到的收缩步长。eta通过缩减特征 的权重使提升计算过程更加保守。缺省值为0.3，取值范围为：[0,1]
        num_class=1,
        nthread=4,
        alpha = 0.05,
        seed=1991)
    print( "training...")
    clf.fit(x_train, y_train[:,i], verbose=False)
    fit_pred = clf.predict_proba(x_val)
    fit_pred1 = clf.predict_proba(x_train)
    print(fit_pred.shape)
    L_test = AUC1(y_val[:,i], fit_pred[:,1])
    X_eicu[:,i] = fit_pred[:,1]
    L_train = AUC1(y_train[:,i], fit_pred1[:,1])
    X_mimic[:, i] = fit_pred1[:, 1]
    print(L_train["roc_auc"], "\n", L_test["roc_auc"])
pd.DataFrame(X_mimic).to_csv(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\model\XGBOOST_mimic_all.csv', index=False)
pd.DataFrame(X_eicu).to_csv(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\model\XGBOOST_eicu_all.csv', index=False)

X_mimic = np.zeros((y_train.shape))
X_eicu = np.zeros((y_val.shape))
for i in range(3):
    clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
            decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
            max_iter=-1, probability=True, random_state=1991, shrinking=True,
            tol=0.001, verbose=False)
    clf.fit(x_train, y_train[:, i])
    fit_pred = clf.predict_proba(x_val)
    fit_pred1 = clf.predict_proba(x_train)
    print(fit_pred.shape)
    L_test = AUC1(y_val[:, i], fit_pred[:, 1])
    X_eicu[:, i] = fit_pred[:, 1]
    L_train = AUC1(y_train[:, i], fit_pred1[:, 1])
    X_mimic[:, i] = fit_pred1[:, 1]
    print(L_train["roc_auc"], "\n", L_test["roc_auc"])
pd.DataFrame(X_mimic).to_csv(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\model\SVM_mimic_all.csv', index=False)
pd.DataFrame(X_eicu).to_csv(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\model\SVM_eicu_all.csv', index=False)

X_mimic = np.zeros((y_train.shape))
X_eicu = np.zeros((y_val.shape))
for i in range(3):
    clf = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                                criterion='gini', max_depth=5, max_features='auto',
                                max_leaf_nodes=None, max_samples=None,
                                min_impurity_decrease=0.0,
                                min_samples_leaf=1, min_samples_split=2,
                                min_weight_fraction_leaf=0.0, n_estimators=30,
                                n_jobs=-1, oob_score=True, random_state=1991,
                                verbose=0, warm_start=False)
    clf.fit(x_train, y_train[:, i])
    fit_pred = clf.predict_proba(x_val)
    fit_pred1 = clf.predict_proba(x_train)
    print(fit_pred.shape)
    L_test = AUC1(y_val[:, i], fit_pred[:, 1])
    X_eicu[:, i] = fit_pred[:, 1]
    L_train = AUC1(y_train[:, i], fit_pred1[:, 1])
    X_mimic[:, i] = fit_pred1[:, 1]
    print(L_train["roc_auc"], "\n", L_test["roc_auc"])
pd.DataFrame(X_mimic).to_csv(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\model\RF_mimic_all.csv', index=False)
pd.DataFrame(X_eicu).to_csv(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\model\RF_eicu_all.csv', index=False)


### The top 20 variables were selected based on the SHAP plot to be screened again for machine learning.
data_mimic = pd.read_csv(r"E:\MIMIC\CCI\result\标准化\论文浓缩版\data\xg2.csv",encoding='utf-8') # mimic
data_eicu = pd.read_csv(r"E:\MIMIC\CCI\result\标准化\论文浓缩版\data\sepsis3_eicu_kmeans_drop_p2.csv",encoding='utf-8') # eicu
Z = np.zeros((20,6))
FFF = [
    ["Mean RDW","Mean AG","Mean RR","Mean HR","Minimum HR","Mean Creatinine","Minimum Temperature","Mean WBC","Maximum SBP","Mean BUN","Mean MCV","Mean DBP","Mean SBP","Mean MAP","Mean FiO2","Minimum Glucose","Minimum PCO2","Mean INR","Mean Temperature","Mean Platelet"],
    ["Mean HR","Mean FiO2","Mean AG","Minimum HR","Minimum RR","Minimum FiO2","Mean RR","Mean RDW","Maximum FiO2","Minimum Glucose","Mean BUN","Mean Base Excess","Mean MCV","Minimum PTT","Minimum SBP","Mean Glucose","Maximum Base Excess","Maximum HR","Maximum Temperature","Mean WBC"],
    ["Minimum PCO2","Mean PO2","Mean MAP","Maximum HR","Mean DBP","Maximum SBP","Maximum FiO2","Mean FiO2","Minimum SBP","Mean RR","Minimum RR","Maximum RR","Maximum Base Excess","Maximum PCO2","Mean PCO2","Maximum Temperature","Mean Glucose","Maximum DBP","Mean HR","Mean Base Excess"]
]
for f in range(1,21):
    print(f)
    n_feature = 1

    from torch import nn
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
        def forward(self, x):
            x = self.net(x)
            return x

    criterion = nn.BCELoss()

    def train(x,y,model1):
        model1.train()
        out1 = model1(x).reshape((-1,n_feature))  # Perform a single forward pass.
        loss = criterion(out1, y.reshape(-1,n_feature))# Compute the loss.
        optim.zero_grad()  # Clear gradients.
        loss.backward()  # Derive gradients.
        optim.step()  # Update parameters based on gradients.
        return AUC(y,out1)

    def test(x,y,model1,pplot=False):
        model1.eval()
        with torch.no_grad():
            out1 = model1(x).reshape((-1,n_feature))
        return AUC(y,out1)  # Derive ratio of correct predictions.


    from sklearn.metrics import roc_curve, auc, confusion_matrix
    from matplotlib import pyplot as plt
    import seaborn as sns
    import scipy

    sns.set(font='SimHei', font_scale=0.8)  # 解决中文问题
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来显示负号

    def AUC(y,out1,pplot=False):
        dict1 = {
            "Precision":[],
            "Recall": [],
            "F1": [],
            "specificity": [],
            "acc": [],
            "roc_auc": [],
            "p": []
        }

        for i in range(n_feature):
            OUT = out1.detach().numpy()[:,i]
            Y = y.detach().numpy().reshape((-1,n_feature))[:,i]
            fpr_BPNN, tpr_BPNN, thresholds_BPNN = roc_curve(Y, OUT)
            dict1["roc_auc"].append(auc(fpr_BPNN, tpr_BPNN))
            stats, p1 = scipy.stats.ranksums(OUT[Y == 1],
                                            OUT[Y == 0], alternative="greater")
            dict1["p"].append(p1)
            C = confusion_matrix(Y, np.round(OUT))
            dict1["Precision"].append(C[1, 1] / (C[1, 1] + C[0, 1]))  # sensitivity
            dict1["Recall"].append(C[1, 1] / (C[1, 1] + C[1, 0]))
            dict1["specificity"].append(C[0, 0] / (C[0, 0] + C[1, 0]))
            dict1["acc"].append((C[0, 0] + C[1, 1]) / sum(sum(C)))
            dict1["F1"].append(2 * dict1["Precision"][-1] * dict1["Recall"][-1] / (dict1["Recall"][-1] + dict1["Precision"][-1]))
            if pplot:
                plt.figure()
                lw = 2
                plt.plot(fpr_BPNN, tpr_BPNN, color='red',
                         lw=lw, label='ROC (AUC_BPNN = %0.4f)' % dict1["roc_auc"][-1])
                plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        return dict1

    from scipy.stats import norm
    def CI(auc, label, alpha = 0.05):
        label = np.array(label)#防止label不是array类型
        n1, n2 = np.sum(label == 1), np.sum(label == 0)
        q1 = auc / (2-auc)
        q2 = (2 * auc ** 2) / (1 + auc)
        se = np.sqrt(abs((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n2 -1) * (q2 - auc ** 2)) / (n1 * n2)))
        confidence_level = 1 - alpha
        z_lower, z_upper = norm.interval(confidence_level)
        se =z_upper * se
        return auc-se,auc+se

    for i in range(3):
        net1 = Model1(f, n_feature)
        optim = torch.optim.Adam(net1.parameters(), lr=0.0015, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00001,
                                 amsgrad=False)  # 优化器：Adam
        feature = FFF[i]
        x_train = data_mimic[feature[:f]].values
        y_train = data_mimic[["hospital_expire_flag", "acute_sepsis", "cci"]].values
        x_val = data_eicu[feature[:f]].values
        y_val = data_eicu[["hosp_mort", "acute_sepsis", "cci"]].values
        x_train = (x_train - x_train.mean()) / x_train.std()
        x_val = (x_val - x_val.mean()) / x_val.std()
        x_train = torch.from_numpy(np.array(x_train, dtype="float32")).reshape((-1, f))
        y_train = torch.from_numpy(np.array(y_train, dtype="float32")).reshape((-1, 3))
        y_train = torch.tensor(y_train, dtype=torch.float32)
        x_val = torch.from_numpy(np.array(x_val, dtype="float32")).reshape((-1, f))
        y_val = torch.from_numpy(np.array(y_val, dtype="float32")).reshape((-1, 3))
        y_val = torch.tensor(y_val, dtype=torch.float32)
        auc_test_max = 0

        for epoch in range(3000):
            # if epoch % 100 == 0:
            #     print(epoch)
            L_train = train(x_train,y_train[:,i],net1)
            L_test = test(x_val,y_val[:,i],net1)
            if np.mean(L_test["roc_auc"]) >auc_test_max:
                auc_test_max = np.mean(L_test["roc_auc"])
                print(np.mean(L_train["roc_auc"]),"\n",np.mean(L_test["roc_auc"]))
                torch.save({'model_1': net1.state_dict()}, r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\model\model_all_{}_{}.pth'.format(f,i))
                Z[f-1,i] = np.array(L_train["roc_auc"])
                Z[f - 1, i+3] = np.array(L_test["roc_auc"])
print(Z)
pd.DataFrame(Z).to_csv(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\model\BPNN_all.csv', index=False)

data_mimic = pd.read_csv(r"E:\MIMIC\CCI\result\标准化\论文浓缩版\data\xg2.csv",encoding='utf-8') # mimic
data_eicu = pd.read_csv(r"E:\MIMIC\CCI\result\标准化\论文浓缩版\data\sepsis3_eicu_kmeans_drop_p2.csv",encoding='utf-8') # eicu
Z = np.zeros((20,6))
FFF = [
    ["Mean RDW","Mean AG","Mean RR","Mean HR","Minimum HR","Mean Creatinine","Minimum Temperature","Mean WBC","Maximum SBP","Mean BUN","Mean MCV","Mean DBP","Mean SBP","Mean MAP","Mean FiO2","Minimum Glucose","Minimum PCO2","Mean INR","Mean Temperature","Mean Platelet"],
    ["Mean HR","Mean FiO2","Mean AG","Minimum HR","Minimum RR","Minimum FiO2","Mean RR","Mean RDW","Maximum FiO2","Minimum Glucose","Mean BUN","Mean Base Excess","Mean MCV","Minimum PTT","Minimum SBP","Mean Glucose","Maximum Base Excess","Maximum HR","Maximum Temperature","Mean WBC"],
    ["Minimum PCO2","Mean PO2","Mean MAP","Maximum HR","Mean DBP","Maximum SBP","Maximum FiO2","Mean FiO2","Minimum SBP","Mean RR","Minimum RR","Maximum RR","Maximum Base Excess","Maximum PCO2","Mean PCO2","Maximum Temperature","Mean Glucose","Maximum DBP","Mean HR","Mean Base Excess"]
]
Y_train = ["hospital_expire_flag", "acute_sepsis", "cci"]
Y_val = ["hosp_mort", "acute_sepsis", "cci"]
for f in range(1,21):
    for i in range(1,2):
        feature = FFF[i]
        x_train = data_mimic[feature[:f]].values
        y_train = data_mimic[Y_train[i]].values
        x_val = data_eicu[feature[:f]].values
        y_val = data_eicu[Y_val[i]].values
        x_train = (x_train - x_train.mean())/x_train.std()
        x_val = (x_val - x_val.mean())/x_val.std()

        clf = xgb.XGBClassifier(
            n_estimators=30,  # 迭代次数
            learning_rate=0.08,  # 步长
            max_depth=5,  # 树的最大深度
            min_child_weight=1,  # 决定最小叶子节点样本权重和
            silent=1,  # 输出运行信息
            subsample=0.8,  # 每个决策树所用的子样本占总样本的比例（作用于样本）
            colsample_bytree=0.8,  # 建立树时对特征随机采样的比例（作用于特征）典型值：0.5-1
            objective='multi:softmax',  # 多分类！！！！！
            eta=0.8, #了防止过拟合，更新过程中用到的收缩步长。eta通过缩减特征 的权重使提升计算过程更加保守。缺省值为0.3，取值范围为：[0,1]
            num_class=2,
            nthread=4,
            seed=1991)
        print( "training...")
        clf.fit(x_train, y_train, verbose=True)
        fit_pred = clf.predict_proba(x_val)
        print(fit_pred.shape)
        L_test = AUC1(y_val, fit_pred[:,1])
        L_train = AUC1(y_train, clf.predict_proba(x_train)[:,1])
        print(np.mean(L_train["roc_auc"]), "\n", np.mean(L_test["roc_auc"]))
        Z[f - 1, i] = np.array(L_train["roc_auc"])
        Z[f - 1, i+3] = np.array(L_test["roc_auc"])

print(Z)
pd.DataFrame(Z).to_csv(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\model\XGOOST_all.csv', index=False)


Z = np.zeros((20,6))
FFF = [
    ["Mean RDW","Mean AG","Mean RR","Mean HR","Minimum HR","Mean Creatinine","Minimum Temperature","Mean WBC","Maximum SBP","Mean BUN","Mean MCV","Mean DBP","Mean SBP","Mean MAP","Mean FiO2","Minimum Glucose","Minimum PCO2","Mean INR","Mean Temperature","Mean Platelet"],
    ["Mean HR","Mean FiO2","Mean AG","Minimum HR","Minimum RR","Minimum FiO2","Mean RR","Mean RDW","Maximum FiO2","Minimum Glucose","Mean BUN","Mean Base Excess","Mean MCV","Minimum PTT","Minimum SBP","Mean Glucose","Maximum Base Excess","Maximum HR","Maximum Temperature","Mean WBC"],
    ["Minimum PCO2","Mean PO2","Mean MAP","Maximum HR","Mean DBP","Maximum SBP","Maximum FiO2","Mean FiO2","Minimum SBP","Mean RR","Minimum RR","Maximum RR","Maximum Base Excess","Maximum PCO2","Mean PCO2","Maximum Temperature","Mean Glucose","Maximum DBP","Mean HR","Mean Base Excess"]
]
Y_train = ["hospital_expire_flag", "acute_sepsis", "cci"]
Y_val = ["hosp_mort", "acute_sepsis", "cci"]
for f in range(1,21):
    for i in range(1,2):
        feature = FFF[i]
        x_train = data_mimic[feature[:f]].values
        y_train = data_mimic[Y_train[i]].values
        x_val = data_eicu[feature[:f]].values
        y_val = data_eicu[Y_val[i]].values
        x_train = (x_train - x_train.mean())/x_train.std()
        x_val = (x_val - x_val.mean())/x_val.std()

        clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
            decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
            max_iter=-1, probability=True, random_state=1991, shrinking=True,
            tol=0.001, verbose=False)
        print( "training...")
        clf.fit(x_train, y_train)
        fit_pred = clf.predict_proba(x_val)
        print(fit_pred.shape)
        L_test = AUC1(y_val, fit_pred[:,1])
        L_train = AUC1(y_train, clf.predict_proba(x_train)[:,1])
        print(np.mean(L_train["roc_auc"]), "\n", np.mean(L_test["roc_auc"]))
        Z[f - 1, i] = np.array(L_train["roc_auc"])
        Z[f - 1, i+3] = np.array(L_test["roc_auc"])

print(Z)
pd.DataFrame(Z).to_csv(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\model\SVM_all.csv', index=False)

Z = np.zeros((20,6))
FFF = [
    ["Mean RDW","Mean AG","Mean RR","Mean HR","Minimum HR","Mean Creatinine","Minimum Temperature","Mean WBC","Maximum SBP","Mean BUN","Mean MCV","Mean DBP","Mean SBP","Mean MAP","Mean FiO2","Minimum Glucose","Minimum PCO2","Mean INR","Mean Temperature","Mean Platelet"],
    ["Mean HR","Mean FiO2","Mean AG","Minimum HR","Minimum RR","Minimum FiO2","Mean RR","Mean RDW","Maximum FiO2","Minimum Glucose","Mean BUN","Mean Base Excess","Mean MCV","Minimum PTT","Minimum SBP","Mean Glucose","Maximum Base Excess","Maximum HR","Maximum Temperature","Mean WBC"],
    ["Minimum PCO2","Mean PO2","Mean MAP","Maximum HR","Mean DBP","Maximum SBP","Maximum FiO2","Mean FiO2","Minimum SBP","Mean RR","Minimum RR","Maximum RR","Maximum Base Excess","Maximum PCO2","Mean PCO2","Maximum Temperature","Mean Glucose","Maximum DBP","Mean HR","Mean Base Excess"]
]
Y_train = ["hospital_expire_flag", "acute_sepsis", "cci"]
Y_val = ["hosp_mort", "acute_sepsis", "cci"]
for f in range(1,21):
    for i in range(1,2):
        feature = FFF[i]
        x_train = data_mimic[feature[:f]].values
        y_train = data_mimic[Y_train[i]].values
        x_val = data_eicu[feature[:f]].values
        y_val = data_eicu[Y_val[i]].values
        x_train = (x_train - x_train.mean())/x_train.std()
        x_val = (x_val - x_val.mean())/x_val.std()

        clf = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                                criterion='gini', max_depth=5, max_features='auto',
                                max_leaf_nodes=None, max_samples=None,
                                min_impurity_decrease=0.0,
                                min_samples_leaf=1, min_samples_split=2,
                                min_weight_fraction_leaf=0.0, n_estimators=30,
                                n_jobs=-1, oob_score=True, random_state=1991,
                                verbose=0, warm_start=False)
        print( "training...")
        clf.fit(x_train, y_train)
        fit_pred = clf.predict_proba(x_val)
        print(fit_pred.shape)
        L_test = AUC1(y_val, fit_pred[:,1])
        L_train = AUC1(y_train, clf.predict_proba(x_train)[:,1])
        print(np.mean(L_train["roc_auc"]), "\n", np.mean(L_test["roc_auc"]))
        Z[f - 1, i] = np.array(L_train["roc_auc"])
        Z[f - 1, i+3] = np.array(L_test["roc_auc"])

print(Z)
pd.DataFrame(Z).to_csv(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\model\RF_all.csv', index=False)