# 设置环境变量 OMP_NUM_THREADS=1。
# 当块数少于可用线程时，KMeans 在使用 MKL 的Windows上会出现内存泄漏。通过设置环境变量 OMP_NUM_THREADS=1 来避免它。
import os
os.environ["OMP_NUM_THREADS"] = '1'
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import v_measure_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

plt.rcParams['font.sans-serif'] = ['SimHei'] # 显示中文
plt.rcParams['axes.unicode_minus'] = False	 # 显示负号

# 计算两向量间的欧氏距离
def distance(X1, X2):
    result = 0
    for (x1, x2) in zip(X1, X2):
        result += (x1 - x2) ** 2
    return np.sqrt(result)

def kmeans(data, k):
    kmeans_model = KMeans(n_clusters=k, n_init='auto', random_state=112)
    kmeans_model.fit(data)
    return (kmeans_model.labels_,
            kmeans_model.cluster_centers_)

def elbow_method(data):
    distortions = []
    K_range = range(1, 11)
    for k in K_range:
        labels, centers = kmeans(data, k)
        distortion = sum(np.min(distance(data[i],centers[labels[i]])**2) for i in range(len(data)))
        distortions.append(distortion)
    plt.plot(K_range, distortions, marker='o')
    plt.xlabel('K值')
    plt.ylabel('簇内离差平方和之和')
    plt.title('手肘法')
    plt.show()

def load_and_process_dataset(dataset):
    # 加载处理数据集
    data = dataset.data
    labels = dataset.target
    #标准差标准化
    MMS = MinMaxScaler().fit(data)
    data = MMS.transform(data)
    # 使用 PCA 降维可视化
    reduced_data = PCA(n_components=2).fit_transform(data)
    return reduced_data, labels

def visualize_clusters(data, labels, centers, title):
    plt.scatter(data[:, 0],data[:, 1],c=labels,cmap='viridis',alpha=0.5)
    plt.scatter(centers[:, 0],centers[:, 1],c='red',marker='X',s=200,label='Centroids')
    plt.title(title)
    plt.legend()
    plt.show()

def visualize_iris(data, labels, title):
    plt.scatter(data[:, 0],data[:, 1],c=labels,cmap='viridis',alpha=0.5)
    plt.title(title)
    plt.show()

# 加载数据集
iris = datasets.load_iris()
# 处理数据集
data, labels = load_and_process_dataset(iris)
# 打印真实标签
label_names = {0: "setosa", 1: "versicolor", 2: "virginica"}
name_labels = [label_names[label] for label in labels]
print(name_labels)
# 可视化数据集
visualize_iris(data, labels, f'Iris')
# 手肘法确定最佳 k 值并打印
elbow_method(data)
#手动选择手肘法得到的最佳 k 值
optimal_k = int(input(f"输入Iris的最佳K值: "))
# 应用 k值聚类，评价指标可视化
Ari, Ami, V_measure, Fmi, Sil, Chi, Dbi = [], [], [], [], [], [], []
for i in range(2,15):
    # 构建聚类模型
    cluster_labels, cluster_centers = kmeans(data, i)
    # 可视化聚类结果
    for j in range(150):
        if cluster_labels[j] == 0:
            cluster_labels[j] = 1
        elif cluster_labels[j] == 1:
            cluster_labels[j] = 0
        else:
            pass
    visualize_clusters(data, cluster_labels, cluster_centers, f'K-means Clustering - Iris')

    if i == optimal_k:
        label_names = {0: "setosa", 1: "versicolor", 2: "virginica"}
        name_labels = [label_names[label] for label in cluster_labels]
        print(name_labels)
    #Ari
    AriScore = adjusted_rand_score(labels, cluster_labels)
    Ari.append(AriScore)
    #Ami
    AmiScore = adjusted_mutual_info_score(labels, cluster_labels)
    Ami.append(AmiScore)
    #V_measure
    V_measure_Score = v_measure_score(labels, cluster_labels)
    V_measure.append(V_measure_Score)
    #Fmi
    FmiScore = fowlkes_mallows_score(labels, cluster_labels)
    Fmi.append(FmiScore)
    #Sil
    Silscore = silhouette_score(data, cluster_labels)
    Sil.append(Silscore)
    #Chi
    ChiScore = calinski_harabasz_score(data, cluster_labels)
    Chi.append(ChiScore)
    #Dbi
    DbiScore = davies_bouldin_score(data, cluster_labels)
    Dbi.append(DbiScore)

name_list = ['ARI', 'AMI', 'V-measure评分', 'FMI', '轮廓系数', '卡林斯基-哈拉巴斯指数', '戴维斯-博尔丁指数']
data_list = [Ari, Ami, V_measure, Fmi, Sil, Chi, Dbi]
for i in range(len(name_list)):
    plt.figure(figsize=(10, 8))
    plt.plot(range(2, 15), data_list[i], linewidth=1.5, linestyle='-')
    plt.xlabel('K值')
    plt.ylabel(f'{name_list[i]}')
    plt.title(f'使用{name_list[i]}评价K-Means聚类模型')
    plt.show()