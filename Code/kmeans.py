import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.metrics import calinski_harabasz_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns

sns.set(font='SimHei',font_scale = 0.6) #解决中文问题
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来显示负号


import numpy as np
def val_sse(x,label,c):
    y = np.zeros(x.values.shape)
    for i in range(x.values.shape[0]):
        y[i,:] = c[label[i]]
    d = (((x.values-y)**2).sum(axis = 1)**(1/2)).mean()
    return d
def sh(n_clusters,X,clusterer,psmx):
    n_clusters = n_clusters
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.3, 0.7])
    ax1.set_ylim([0, X.shape[0] + (n_clusters + 1) * 10])
    # clusterer = KMeans(n_clusters=n_clusters, random_state=10).fit(X)
    cluster_labels = clusterer.predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    y_lower = 10
    C = ["A", "B", "C", "D", "E", "F", "G"]
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper)
                          , ith_cluster_silhouette_values
                          , facecolor=color
                          , alpha=0.7
                          )
        ax1.text(-0.05
                 , y_lower + 0.5 * size_cluster_i
                 , C[i])
        y_lower = y_upper + 10
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    # 创建PCA对象，设置降维后的维度
    pca = PCA(n_components=2)
    pca.fit(psmx)
    reduced_data = pca.transform(X)
    centers = clusterer.cluster_centers_
    reduced_center = pca.transform(centers)
    ax2.scatter(reduced_data[:, 0], reduced_data[:, 1]
                , marker='o'
                , s=8
                , c=colors
                )

    # Draw white circles at cluster centers
    ax2.scatter(reduced_center[:, 0], reduced_center[:, 1], marker='x',
                c="red", alpha=1, s=200)

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")
    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')
    plt.savefig(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\kmeans\{}聚类结果_mimic.tif'.format(n_clusters), dpi=300)
    plt.show()
def sh1(n_clusters,X,clusterer,psmx):
    n_clusters = n_clusters
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.3, 0.7])
    ax1.set_ylim([0, X.shape[0] + (n_clusters + 1) * 10])
    # clusterer = KMeans(n_clusters=n_clusters, random_state=10).fit(X)
    cluster_labels = clusterer.predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    y_lower = 10
    # C = ["D","E","A","C","B","F","G"]
    C = ["A", "B", "C", "D", "E", "F", "G"]
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper)
                          , ith_cluster_silhouette_values
                          , facecolor=color
                          , alpha=0.7
                          )
        ax1.text(-0.05
                 , y_lower + 0.5 * size_cluster_i
                 , C[i])
        y_lower = y_upper + 10
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    # 创建PCA对象，设置降维后的维度
    pca = PCA(n_components=2)
    pca.fit(psmx)
    reduced_data = pca.transform(X)
    centers = clusterer.cluster_centers_
    reduced_center = pca.transform(centers)
    ax2.scatter(reduced_data[:, 0], reduced_data[:, 1]
                , marker='o'
                , s=8
                , c=colors
                )

    # Draw white circles at cluster centers
    ax2.scatter(reduced_center[:, 0], reduced_center[:, 1], marker='x',
                c="red", alpha=1, s=200)

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")
    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')
    plt.savefig(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\kmeans\{}聚类结果_eicu.tif'.format(n_clusters), dpi=300)
    plt.show()




data = pd.read_csv(r"E:\MIMIC\CCI\result\标准化\论文浓缩版\data\xg2.csv",encoding='utf-8') # mimic
data1 = pd.read_csv(r"E:\MIMIC\CCI\result\标准化\论文浓缩版\data\sepsis3_eicu_kmeans_drop_p2.csv",encoding='utf-8') # eicu

x = data.iloc[:,17:78]
# x2 = x.copy()
# x2['glucose_avg'] = x2['glucose_avg']/18
# x2['glucose_min'] = x2['glucose_min']/18
x2 = data1.iloc[:,21:82]
x1 = x.copy()
for i in range(61):
    mean_value = x.iloc[:, i].mean()
    sd_value = x.iloc[:, i].std()
    x.iloc[x.iloc[:,i]>(mean_value+20*sd_value),i] = mean_value+20*sd_value
    x.iloc[x.iloc[:, i] < (mean_value - 20 * sd_value), i] = mean_value - 20 * sd_value
for i in range(61):
    mean_value = x2.iloc[:, i].mean()
    sd_value = x2.iloc[:, i].std()
    x2.iloc[x2.iloc[:, i] > (mean_value + 20 * sd_value), i] = mean_value + 20 * sd_value
    x2.iloc[x2.iloc[:, i] < (mean_value - 20 * sd_value), i] = mean_value - 20 * sd_value


x_norm = (x - x.mean()) / (x.std())
x2= (x2 - x2.mean()) / (x2.std())


X_train, X_test= train_test_split(x_norm, test_size=0.3,random_state=1)
# val_sse(X_train,np.array(np.zeros(X_train.values.shape[0]),dtype = np.int8),X_train.values.mean(0).reshape((1,-1)))
SSE_train=[]
SSE_val = []
d_sc_train=[]
d_sc_val = []
SSE=[]
sc_score_train = []
sc_score_val = []
CH_train =[]
CH_val =[]
K = []
for i in range(2,8):
    cluster_smallsub = KMeans(n_clusters = i, max_iter=800,random_state = 10).fit(X_train)
    y_test = cluster_smallsub.predict(X_test)
    centroid = cluster_smallsub.cluster_centers_
    SSE.append(cluster_smallsub.inertia_)
    SSE_train.append(val_sse(X_train,cluster_smallsub.labels_,centroid))
    SSE_val.append(val_sse(X_test,y_test,centroid))
    sc_score_train.append(metrics.silhouette_score(X_train, cluster_smallsub.labels_))
    sc_score_val.append(metrics.silhouette_score(X_test, y_test))
    CH_train.append(calinski_harabasz_score(X_train, cluster_smallsub.labels_))
    CH_val.append(calinski_harabasz_score(X_test, y_test))
    if i >2:
        d_sc_train.append(sc_score_train[-1]-sc_score_train[-2])
        d_sc_val.append(sc_score_val[-1]-sc_score_val[-2])
    K.append(i)
    sh(i,x_norm.values,cluster_smallsub,x2.values)
    sh1(i, x2.values, cluster_smallsub,x2.values)

plt.plot(K,SSE,color='red',linewidth=2.0,linestyle='--',marker='o')
plt.savefig(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\kmeans\簇内平方和.tif', dpi=300)
plt.show()

plt.plot(K,SSE_train,color='red',linewidth=2.0,linestyle='--',marker='o')
plt.plot(K,SSE_val,color='blue',linewidth=2.0,linestyle='--',marker='o')
plt.ylabel("SSE")
plt.savefig(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\kmeans\SSE.tif', dpi=300)
plt.show()

plt.plot(K[1:],d_sc_train,color='red',linewidth=2.0,linestyle='--',marker='o')
plt.plot(K[1:],d_sc_val,color='blue',linewidth=2.0,linestyle='--',marker='o')
plt.ylabel("d_SSE")
plt.savefig(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\kmeans\d_sc_score.tif', dpi=300)
plt.show()

plt.plot(K,sc_score_train,color='red',linewidth=2.0,linestyle='--',marker='o')
plt.plot(K,sc_score_val,color='blue',linewidth=2.0,linestyle='--',marker='o')
plt.ylabel("silhouette")
plt.savefig(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\kmeans\轮廓系数.tif', dpi=300)
plt.show()

plt.plot(K,CH_train,color='red',linewidth=2.0,linestyle='--',marker='o')
plt.plot(K,CH_val,color='blue',linewidth=2.0,linestyle='--',marker='o')
plt.ylabel("Calinski-Harabaz")
plt.savefig(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\kmeans\Calinski-Harabaz系数.tif', dpi=300)
plt.show()

cluster_smallsub = KMeans(n_clusters = 4, max_iter=800,random_state = 10).fit(X_train)
data1["kmeans"] = cluster_smallsub.predict(x2)
data["kmeans"] = cluster_smallsub.predict(x_norm)
data1.to_csv(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\data\sepsis3_eicu_kmeans_drop_p2.csv', index=False) # eicu
data.to_csv(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\data\xg2.csv', index=False) # mimic

# # #
label_t = cluster_smallsub.predict(x_norm) # MIMIC
centroid = cluster_smallsub.cluster_centers_
print(x_norm.values.shape)
xx = np.zeros((61,5))
for i in range(61):
    x_copy = x_norm.copy() # MIMIC
    for j in range(4):
        x_copy.iloc[label_t==j,i]=centroid[j,i]
    label_f = cluster_smallsub.predict(x_copy)
    xx[i,0] = (label_t!=label_f).sum()/label_f.sum()
    for j in range(4):
        xx[i,j+1] = (label_f[label_t==j]!=j).sum()/(label_t==j).sum()
print(xx)
print(x_norm.columns)
df1=pd.DataFrame(xx,index=x_norm.columns,columns=["all","A","B","C","D"])
df1.to_csv(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\data\importance_feature_mimic.csv')

label_t = cluster_smallsub.predict(x2) # EICU
centroid = cluster_smallsub.cluster_centers_
print(x_norm.values.shape)
xx = np.zeros((61,5))
for i in range(61):
    x_copy = x2.copy() # EICU
    for j in range(4):
        x_copy.iloc[label_t==j,i]=centroid[j,i]
    label_f = cluster_smallsub.predict(x_copy)
    xx[i,0] = (label_t!=label_f).sum()/label_f.sum()
    for j in range(4):
        xx[i,j+1] = (label_f[label_t==j]!=j).sum()/(label_t==j).sum()
print(xx)
print(x_norm.columns)
df1=pd.DataFrame(xx,index=x_norm.columns,columns=["all","A","B","C","D"])
df1.to_csv(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\data\importance_feature_eicu.csv')

# Notice: running a CDF curve may take 4-8h!
# # CDF
# nn = x_norm.values.shape[0] # MIMIC
# # nn = x2.values.shape[0] # EICU
# n = 40
# k_nn = 7
# S_cdf = []
# Color = ["darkorange","orangered","gold","yellowgreen","c","m"]
# for i in range(2,k_nn+1):
#     XX = np.zeros((nn,nn))
#     for num in range(n):
#         print(num)
#         X_train, X_test = train_test_split(x_norm, test_size=0.3, random_state=num)
#         cluster_smallsub = KMeans(n_clusters=i, max_iter=800, random_state=10).fit(X_train)
#         label = cluster_smallsub.predict(x_norm).reshape(1,-1) # MIMIC
#         # label = cluster_smallsub.predict(x2).reshape(1, -1) # EICU
#         XX[(label-label.T)==0] += 1
#
#     XX = XX/n
#
#     XX_upper_no_diag = XX[np.triu_indices(nn, k = 1)]
#     cdf = []
#     for c in np.arange(0,1+1/n,1/n):
#         cdf.append((XX_upper_no_diag<=c).sum()/(nn*(nn-1)/2))
#     cdf = np.array(cdf)
#     S_cdf.append((cdf*1/n).sum())
#     print(S_cdf[-1])
#     colors = Color[i-2]
#     lab = "K = "+str(i)
#     print(lab)
#     plt.plot(np.arange(0,1+1/n,1/n),cdf,color=colors,linewidth=2.0)
# plt.yticks([0.2, 0.4, 0.6, 0.8, 1])
# plt.xticks([ 0, 0.2, 0.4, 0.6, 0.8, 1])
# plt.legend(("K = 2","K = 3","K = 4","K = 5","K = 6","K = 7"))
#
# plt.ylabel("CDF")
# plt.xlabel("Consensus Index")
# plt.savefig(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\kmeans\CDF_mimic.tif', dpi=300)
# # plt.savefig(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\kmeans\CDF_eicu.tif', dpi=300)
# plt.yticks([0.2, 0.4, 0.6, 0.8, 1])
# plt.grid(False)  ##关闭网格
# plt.show()
#
# plt.plot(range(2,k_nn+1),S_cdf,color="red",linewidth=2.0,linestyle='--',marker='o')
# plt.yticks([0.2, 0.4, 0.6, 0.8, 1])
# plt.ylabel("Area under CDF curve")
# plt.xlabel("K")
# plt.savefig('r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\kmeans\S_CDF_mimic.tif', dpi=300)
# # plt.savefig('r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\kmeans\S_CDF_eicu.tif', dpi=300)
# plt.show()
#
# d_cdf = []
# for i in range(6):
#     if i == 0:
#         d_cdf.append(S_cdf[0])
#     else:
#         d_cdf.append((S_cdf[i]-S_cdf[i-1])/S_cdf[i-1])
# plt.plot(range(2,8),d_cdf,color="red",linewidth=2.0,linestyle='--',marker='o')
# plt.yticks([0,0.2, 0.4, 0.6])
# plt.xticks([2,3,4,5,6,7])
# plt.ylabel("relative change in the area under CDF curve")
# plt.xlabel("K")
# plt.savefig(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\kmeans\d_CDF_mimic.tif', dpi=300) # MIMIC
# # plt.savefig(r'E:\MIMIC\CCI\result\标准化\论文浓缩版\result\kmeans\d_CDF_eicu.tif', dpi=300) # EICU
# plt.show()
#
#
from numpy.random import uniform
from scipy.spatial.distance import cdist
#
#
# n维霍普金斯统计量计算，input:DataFrame类型的二维数据，output:float类型的霍普金斯统计量
# 默认从数据集中抽样的比例为0.3
def hopkins_statistic(data: pd.DataFrame, sampling_ratio: float = 0.3) -> float:
    # 抽样比例超过0.1到0.5区间任意一端则用端点值代替
    sampling_ratio = min(max(sampling_ratio, 0.1), 0.5)
    # 抽样数量
    n_samples = int(data.shape[0] * sampling_ratio)
    # 原始数据中抽取的样本数据
    sample_data = data.sample(n_samples)
    # 原始数据抽样后剩余的数据
    data = data.drop(index=sample_data.index)  # ,inplace = True)
    # 原始数据中抽取的样本与最近邻的距离之和
    data_dist = cdist(data, sample_data).min(axis=0).sum()
    # 人工生成的样本点，从平均分布中抽样(artificial generate samples)
    ags_data = pd.DataFrame({col: uniform(data[col].min(), data[col].max(), n_samples) \
                             for col in data})
    # 人工样本与最近邻的距离之和
    ags_dist = cdist(data, ags_data).min(axis=0).sum()
    # 计算霍普金斯统计量H
    H_value = ags_dist / (data_dist + ags_dist)
    return H_value

H_value = []
for i in range(10):
    H = hopkins_statistic(x_norm) # MIMIC
    H_value.append(H)
    print(H)
print("霍普金斯统计量: MIMIC")
print(np.mean(H_value))

H_value = []
for i in range(10):
    H = hopkins_statistic(x2) # EICU
    H_value.append(H)
    print(H)
print("霍普金斯统计量: eICU")
print(np.mean(H_value))