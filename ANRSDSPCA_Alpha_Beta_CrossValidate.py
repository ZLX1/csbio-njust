import numpy as np
import pandas as pd
import warnings
import os
import time
import operator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from sklearn.preprocessing import normalize
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")


def rbf(dist, t):
    '''
    rbf kernel function
    '''
    return np.exp(-(dist / t))

def cal_pairwise_dist(x):
    '''计算pairwise 距离, x是matrix
    (a-b)^2 = a^2 + b^2 - 2*a*b
    '''
    # sum_x = np.sum(np.square(x), 1)
    # dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    # 返回任意两个点之间距离的平方
    distlist = pdist(x, metric='euclidean')
    dist = squareform(distlist)
    return dist

def cal_rbf_dist(data, n_neighbors, t):
    dist = cal_pairwise_dist(data)
    n = dist.shape[0]
    rbf_dist = rbf(dist, t)
    W_L = np.zeros((n, n))
    W_G = np.zeros((n, n))
    for i in range(n):
        index_L = np.argsort(dist[i])[1:1 + n_neighbors]
        W_L[i, index_L] = rbf_dist[i, index_L]
        W_L[index_L, i] = rbf_dist[index_L, i]

        index_G = np.argsort(dist[i])[1 + n_neighbors:]
        W_G[i, index_G] = rbf_dist[i, index_G]
        W_G[index_G, i] = rbf_dist[index_G, i]
    return W_L,W_G

def cal_laplace(data1,data2):
    N = data1.shape[0]
    # W_L,W_G = cal_rbf_dist(data, n_neighbors, t)  # 建立邻接矩阵W，参数有最近k个邻接点，以及热核参数t
    D_L = np.zeros_like(data1)
    D_G = np.zeros_like(data2)

    for i in range(N):
        D_L[i, i] = np.sum(data1[i])  # 求和每一行的元素的值，作为对角矩阵D的对角元素
        D_G[i, i] = np.sum(data2[i])
    L_L = D_L - data1  # L矩阵
    L_G = D_G - data2
    return L_L,L_G

def cal_Local_global_laplace(data):
    N = data.shape[0]
    H = np.zeros_like(data)
    for i in range(N):
        H[i, i] = np.sum(data[i])  # 求和每一行的元素的值，作为对角矩阵D的对角元素
    Local_global_L = H - data  # L矩阵
    return Local_global_L

def ANRSDSPCA_Algorithm(xMat,bMat,laplace,alpha,beta,gamma,k,c,n):
    obj1 = 0
    obj2 = 0
    thresh = 1e-50
    A = np.random.rand(c, k)  # (4,3)
    V = np.eye(n)  # (500, 500)
    vMat = np.mat(V)  # (500, 500)
    for m in range(0, 10):
        # print('xMat = ',xMat)
        # print('xMat.shape = ', xMat.shape)
        # print('bMat = ', bMat)
        # print('bMat.shape = ', bMat.shape)
        Z = -(xMat.T * xMat) - (alpha * bMat.T * bMat) + beta * vMat + gamma * laplace  # (643, 643)
        # Z = -(xMat.T * xMat) - (alpha * bMat.T * bMat) + beta * vMat  # (643, 643)
        # 计算Q
        Z_eigVals, Z_eigVects = np.linalg.eig(np.mat(Z))
        # 对特征值从小到大排序
        eigValIndice = np.argsort(Z_eigVals)
        # 最小的k个特征值的下标,
        # k表示降维的个数
        n_eigValIndice = eigValIndice[0:k]
        # 最小的k个特征值对应的特征向量
        n_Z_eigVect = Z_eigVects[:, n_eigValIndice]
        Q = np.array(n_Z_eigVect)  # (643, 3)
        # 计算V
        # 计算Q的行二范数
        q = np.linalg.norm(Q, ord=2, axis=1) + 1e-6
        qq = 1.0 / (q * 2)
        VV = np.diag(qq)  # (643, 643)
        vMat = np.mat(VV)  # (643, 643)
        qMat = np.mat(Q)  # (643, 3)
        # 计算Y
        Y = xMat * qMat  # (20502, 3)
        # 计算A
        A = bMat * qMat  # (4, 3)

        obj1 = (np.linalg.norm(xMat - Y * qMat.T, ord='fro')) ** 2 + alpha * (
            np.linalg.norm(bMat - A * qMat.T, ord='fro')) ** 2 + beta * np.trace(qMat.T * vMat * qMat) + gamma * np.trace(qMat.T * laplace * qMat)
        if m > 0:
            diff = obj2 - obj1
            if diff < thresh:
                break
        obj2 = obj1
    return Y,qMat


def cal_projections(X_data,B_data,alpha1,beta1,gamma1,k_d):
    nclass = 4
    n = len(X_data)  # 500
    dist = cal_pairwise_dist(X_data)
    max_dist = np.max(dist)
    W_L, W_G = cal_rbf_dist(X_data, n_neighbors=5, t=max_dist)
    L_L, L_G = cal_laplace(W_L, W_G)
    # L_L最大特征值---谱半径
    L_L_eigVals = max(np.linalg.eig(L_L)[0])
    # L_G最大特征值---谱半径
    L_G_eigVals = max(np.linalg.eig(L_G)[0])
    # 决定Eta值
    Eta = L_G_eigVals / (L_G_eigVals + L_L_eigVals)
    # Eta = 0.5
    # 总的局部信息和全局信息之和
    R = Eta * W_L - (1 - Eta) * W_G
    # 局部信息和全局信息之和的拉普拉斯矩阵
    M = cal_Local_global_laplace(R)
    Y, Q = ANRSDSPCA_Algorithm(X_data.transpose(), B_data.transpose(), M, alpha1, beta1, gamma1, k_d, nclass, n)
    return Y,Q


if __name__ == '__main__':
    X_filepath = 'D:\\MachineLearning\\Python\\pyCharmProjects\\ML\\csbio\\data\\X_original_GAI.csv'
    X_original = pd.read_csv(X_filepath)#(20502, 643)
    X_original = X_original.values#(20502, 643)
    sc = MinMaxScaler()
    X_original = sc.fit_transform(X_original)
    Y_filepath = 'D:\\MachineLearning\\Python\\pyCharmProjects\\ML\\csbio\\data\\gnd4class_4_GAI.csv'
    Y_gnd4class4 = pd.read_csv(Y_filepath)#(643, 4)
    Y_gnd4class4 = Y_gnd4class4.values.transpose()#(4, 643)
    X = np.mat(X_original)#(20502, 643)
    B = np.mat(Y_gnd4class4)#(4, 643)
    nclass = 4
    k_d = 4
    count = 0
    correctlist = []
    correctlist.clear()
    x_train, x_test, y_train, y_test = train_test_split(X.T, B.T, test_size=143, random_state=1)
    knc = KNeighborsClassifier(n_neighbors=5)
    # 5折交叉验证
    KF = KFold(n_splits=5)
    for alpha in np.logspace(-10,10,21):
        for beta in np.logspace(-10,10,21):
            # correct = 0
            print('count = ',count)
            count += 1
            per_correct = 0
            for train_index, validation_index in KF.split(x_train):
                x_train_t = x_train[train_index]
                x_train_v = x_train[validation_index]
                y_train_t = y_train[train_index]
                y_train_v = y_train[validation_index]
                Y_train_proj, Q_train_proj = cal_projections(x_train_t, y_train_t, alpha, beta, 0, k_d)
                Y_train_proj = np.mat(Y_train_proj)
                Y_train_proj = (((Y_train_proj.T * Y_train_proj).I) * (Y_train_proj.T)).T
                # Q_train_proj1 = (np.mat(x_train_t) * Y_train_proj)
                Q_test_proj = (np.mat(x_train_v) * Y_train_proj)  # 100，4
                # Y_test_proj, Q_test_proj = cal_projections(x_train_v, y_train_v, alpha, beta, 0, k_d)
                knc.fit(np.real(Q_train_proj), y_train_t)
                y_predict = knc.predict(np.real(Q_test_proj))
                per_correct += knc.score(np.real(Q_test_proj), y_train_v)
            print('交叉验证准确率：', per_correct / 5)
            correctlist.append(per_correct / 5)
    mean_correct_rate = np.array(correctlist).reshape(21,21)
    mean_correct_rate_PD = pd.DataFrame(mean_correct_rate)
    datapath1 = 'D:\\MachineLearning\\correct_a_b_k4.csv'
    mean_correct_rate_PD.to_csv(datapath1, index=False)