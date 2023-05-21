'''
用于保存用于监督哈希绘制指标的算法库
分别为 ： PR曲线 （1） 每个哈希码长度下所有查询图像的平均 PR 曲线,适用于评估不同哈希码长度的整体效果
        PR曲线 （2） 每个查询图像在不同哈希码长度下的 PR 曲线,再求平均,适用于分析不同查询图像的哈希长度敏感度
        TopK-Precision : 计算前K个中的正样本占比正负样本的比例
        NDGC ： 评价正样本是否更好排序的指标

'''

import matplotlib
matplotlib.use('WebAgg')
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as scio
import os

def calculate_ndcg(hashed_retrieval_set, hashed_query_set, retrieval_labels,query_labels, k=1000):
    """
    计算NDCG@k
    参数:
        hashed_retrieval_set: ndarray, 检索集的哈希码
        hashed_query_set: ndarray, 查询集的哈希码
        retrieval_labels: ndarray, 检索集的标签
        query_labels: ndarray, 查询集的标签
        k: int, NDCG@k中的k值 (默认值为1000)
    返回:
        ndcg: float, NDCG@k的值
    """
    def similarity_score(h1, h2):
        """
        计算哈希码之间的相似度分数
        """
        return np.sum(h1 == h2)
    def dcg(relevance_scores, k):
        """
        计算DCG
        """
        return np.sum([(2**relevance_scores[i] - 1) / np.log2(i + 2) for i in range(min(k, len(relevance_scores)))])

    def idcg(relevance_scores, k):
        """
        计算IDCG
        """
        sorted_scores = np.sort(relevance_scores)[::-1]
        return dcg(sorted_scores, k)
    ndcg_values = []
    for query_index, query_hash in enumerate(hashed_query_set):
        query_label = query_labels[query_index]
        # 计算查询与检索集的相似度分数
        scores = [similarity_score(query_hash, retrieval_hash) for retrieval_hash in hashed_retrieval_set]
        # 计算相关性评分
        relevance_scores = np.array([1 if query_label == retrieval_label else 0 for retrieval_label in retrieval_labels])
        # 计算NDCG
        dcg_value = dcg(relevance_scores, k)
        idcg_value = idcg(relevance_scores, k)
        if idcg_value == 0:
            ndcg_values.append(0)
        else:
            ndcg_values.append(dcg_value / idcg_value)
        return np.mean(ndcg_values)


def topk_recall(qF, rF, qL, rL, k=100):
    # qF: query feature, tensor of size (num_query, code_length)
    # rF: retrieval feature, tensor of size (num_retrieval, code_length)
    # qL: query label, tensor of size (num_query,)
    # rL: retrieval label, tensor of size (num_retrieval,)
    # k: top k retrieval

    num_query = qF.shape[0]

    # 计算相似度矩阵
    sim_mat = qF @ rF.T

    # 按相似度排序
    rank_mat = np.argsort(-sim_mat, axis=1)

    # 计算top-K recall
    recall = np.zeros(num_query)
    for i in range(num_query):
        pos_idx = np.argwhere(rL==qL[i]).reshape(-1)
        recall[i] = (np.isin(pos_idx, rank_mat[i, :k])).sum() / pos_idx.size

    print(f'top-{k} Recall: {recall.mean()}')
    return recall.mean()


def one_hot(ar , num = 10 , np_type = False):
    "输入你要onehot的变量，然后输入你onehot编码的长度，默认返回pytorch张量"
    ar = np.squeeze(ar)
    w = np.zeros((ar.shape[0] , num))
    print(w.shape)
    for i in range(ar.shape[0]):
        w[i][ar[i]] = 1
    if np_type :
        return w
    else:
        return torch.from_numpy(w)
def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH

def calculate_ndcg(hashed_retrieval_set, hashed_query_set, retrieval_labels, query_labels, k=1000):
    """
    计算NDCG@k
    参数:
        hashed_retrieval_set: ndarray, 检索集的哈希码
        hashed_query_set: ndarray, 查询集的哈希码
        retrieval_labels: ndarray, 检索集的标签
        query_labels: ndarray, 查询集的标签
        k: int, NDCG@k中的k值 (默认值为1000)
    返回:
        ndcg: float, NDCG@k的值
    """
    def similarity_score(h1, h2):
        """
        计算哈希码之间的相似度分数
        """
        return np.sum(h1 == h2)
    def dcg(relevance_scores, k):
        """
        计算DCG
        """
        return np.sum([(2**relevance_scores[i] - 1) / np.log2(i + 2) for i in range(min(k, len(relevance_scores)))])
    def idcg(relevance_scores, k):
        """
        计算IDCG
        """
        sorted_scores = np.sort(relevance_scores)[::-1]
        return dcg(sorted_scores, k)
    ndcg_values = []
    for query_index, query_hash in enumerate(hashed_query_set):
        query_label = query_labels[query_index]
        # 计算查询与检索集的相似度分数
        scores = [similarity_score(query_hash, retrieval_hash) for retrieval_hash in hashed_retrieval_set]
        # 计算相关性评分
        relevance_scores = np.array([1 if query_label == retrieval_label else 0 for retrieval_label in retrieval_labels])
        # 计算NDCG
        dcg_value = dcg(relevance_scores, k)
        idcg_value = idcg(relevance_scores, k)
        if idcg_value == 0:
            ndcg_values.append(0)
        else:
            ndcg_values.append(dcg_value / idcg_value)
    return np.mean(ndcg_values)

def PR_curve_focus_on_binary_length(qB, rB, query_label, retrieval_label,is_exist_minus_one = True): # 只接受0和1的哈希码值
    "只接受二进制码组成的0和1 或者 -1和1组成的二进制码，请把所有标签进行onehot编码，传入torch张量"
    help(PR_curve_focus_on_binary_length)
    if is_exist_minus_one:
        qB[qB==-1] = 0 ; rB[rB==-1] = 0 # 把【-1,1】组成的哈希码变成【0,1】
    num_query = qB.shape[0]
    num_bit = qB.shape[1]
    P = torch.zeros(num_query, num_bit+1)
    R = torch.zeros(num_query, num_bit+1)
    for i in range(num_query):
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        # print('gnd size is ' + str(gnd.shape))
        '''
        对于一个(num,4)与(4,num)大小的矩阵相乘最后得到是(num,num)大小的矩阵，他会展示标签之间是否相同
        # 这里的groudtruth标签大小是（retrieval_label,)
        '''
        tsum = torch.sum(gnd) # 一共有多少个相同的标签（事实上的，没有固定检索哈希码的长度）
        if tsum == 0:
            continue # 如果没有相似的也没必要继续进行了
        hamm = calc_hamming_dist(qB[i, :], rB) # 汉明距离
        # print('hamming distance is '+str(hamm.shape))
        tmp = (hamm <= torch.arange(0, num_bit+1).reshape(-1, 1).float().to(hamm.device)).float()
        # print('tmp size is '+str(tmp.shape))
        '''
        计算的汉明距离与特定哈希码长度做比较，如果超出了当前哈希码长度就是False
        解释：取一个查询集的哈希码，计算与所有检索集的汉明距离，然后使用tmp变量来存储特定哈希码长度下（有最大阈值的限制）的表现 
            tmp产生的列表是(num_bits + 1 , retrieval_label) 要么是True要么是False
        '''
        # 在指定哈希码长度下能检测到的最大阈值——哈希码的长度是多少 （ 也就是在特定哈希码长度的限制之下，能检测到的最多的检索集个数
        total = tmp.sum(dim=-1) # 降维1维列表,对应的每个哈希码长度对应的检索正确的整数
        # print('total size is'+str(total.shape))
        total = total + (total == 0).float() * 0.1 # 将为0的元素变成0.1
        t = tmp * gnd # t是能被当前哈希码长度准确检测的到的实际个数 shape(num_bits,) —— 正确预测
        # print('t size is '+str(t.shape))
        count = t.sum(dim=-1) # 所有可以被当前某个特定哈希码长度有效检索的图像的总数
        '''
        强调一下，这里的变量所代表的意义 ： 
        1 ， count 由 t 产生 代表当前哈希码长度下，能正确匹配到的正样本数量
        2 ， t是能被当前哈希码长度正确检测的实际个数 ， t 是比gnd小的 ，因为它更代表能被当前哈希码检索的这个前提条件还能识别到的groud truth标签
        3 ， total 对应每个特定哈希码长度下所能匹配到的最大匹配数量，包含了正样本和负样本
        4 ， tsum 本来就能就是正确的总数（正确答案）也就是所有的groud truth标签
        '''
        p = count / total # percision（精度） = （所有特定哈希码长度下可以有效检索的个数）/ (每个特定哈希码下正确的个数） —— 正确预测 / 总样本数 —— 模型的整体性能
        # print('p shape is ' + str(p.shape))

        r = count / tsum # 正确预测 / 所有的正确样本
        # print('r shape is ' + str(r.shape))
        P[i] = p # 第一个查询标签对应的Persion的指标
        R[i] = r # 以此类推
    # print(f'P size is {str(P.shape)}')
    # print(f'R size is {str(R.shape)}')
    mask = (P > 0).float().sum(dim=0) # 只考虑含有正样本的查询集
    mask = mask + (mask == 0).float() * 0.1
    P = P.sum(dim=0) / mask
    R = R.sum(dim=0) / mask
    '''
    num_bit表示不同的哈希码长度,范围为0到num_bit
    对于每个query,计算其在不同哈希码长度下的precision(P)和recall(R)
    计算完成后,利用mask只考虑非零值,求出所有query在每个哈希码长度下的平均P和平均R
    '''
    # plt.plot(R, P, linestyle="-", marker='D', color='blue',label = 'DSH')
    # plt.text(0.5, -0.1, '(a) PR curve @ 16bits', ha='center', va='center', fontsize=16, fontweight='bold', transform=plt.gca().transAxes)
    # plt.grid(True)
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # # 调整图像的大小
    # fig = plt.gcf()
    # fig.set_size_inches(9,9)
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.legend()  # 加图例
    # plt.show()

    return P, R

def save_csv(filename , v1 , v2 ,v1_name , v2_name):
    final = '.csv'
    os.makedirs(os.path.join(filename) , exist_ok=True)
    np.savetxt(os.path.join(filename ,v1_name + final) , v1)
    np.savetxt(os.path.join(filename ,v2_name + final) , v2)

def PR_curve_focus_on_retrieval_img(query_binary, retrieval_binary, query_label, retrieval_label):
    "No need to onehot the label ， 投入numpy数组"
    """
    绘制检索评价的精度-召回率曲线（PR曲线）
    Args:
        query_binary (numpy.ndarray): 一个大小为 (num_query, num_bit) 的 numpy 数组，
            包含查询图像的二进制哈希码。
        retrieval_binary (numpy.ndarray): 一个大小为 (num_retrieval, num_bit) 的 numpy 数组，
            包含检索图像的二进制哈希码。
        query_label (numpy.ndarray): 一个大小为 (num_query,) 的 numpy 数组，
            包含查询图像的真实标签。
        retrieval_label (numpy.ndarray): 一个大小为 (num_retrieval,) 的 numpy 数组，
            包含检索图像的真实标签。

   
    """
    # Convert the labels to int32 type to avoid indexing errors
    ## 将标签转换为 int32 类型，以避免索引错误
    query_label = query_label.astype(np.int32)
    retrieval_label = retrieval_label.astype(np.int32)

    # Calculate the Hamming distances between query and retrieval binary codes
    # 计算查询图像和检索图像之间的汉明距离
    hamming_dist = np.count_nonzero(query_binary[:, np.newaxis, :]
                                    != retrieval_binary[np.newaxis, :, :], axis=2)
    '''
    query_binary 和 retrieval_binary 是二进制向量,形状分别是(m, n) 和(p, n),n 为向量维度。
    query_binary[:, np.newaxis, :] 将查询向量形状改为(m, 1, n),插入第2维一个1维度。
    retrieval_binary[np.newaxis, :, :] 将索引向量形状改为(1, p, n),插入第1维一个1维度。
    然后使用 != 计算这两个三维矩阵对应位置不相等的元素个数,结果形状是(m, p)。
    np.count_nonzero() 计算Axis=2 不为0 的元素个数,即每个二维的(m, p) 对应位置的汉明距离。
    '''
    print(hamming_dist)
    # Sort the retrieval samples by ascending order of Hamming distance
    # 将检索图像按汉明距离排序
    idx = np.argsort(hamming_dist, axis=1) # 按照索引进行排序，汉明距离最小的索引派在前面

    # Initialize the precision-recall arrays
    # 初始化精度和召回率数组
    num_query = query_binary.shape[0]
    num_retrieval = retrieval_binary.shape[0]
    precision = np.zeros((num_query, num_retrieval))
    recall = np.zeros((num_query, num_retrieval))

    # Compute the precision-recall values for each query sample
    # 计算每个查询图像的精度和召回率
    for i in range(num_query):
        # Compute the ground-truth labels for the retrieval samples
        # 计算检索图像的真实标签
        gnd = (query_label[i] == retrieval_label[idx[i]])
        # Compute the cumulative sums of true positives and false positives
        # 计算真正例和假正例的累积和
        tp_cumsum = np.cumsum(gnd)
        fp_cumsum = np.cumsum(~gnd) # ～表示补码 ， 取反
        # Compute the precision and recall values
        # 计算精度和召回率
        precision[i] = tp_cumsum / (tp_cumsum + fp_cumsum) # 正样本占总样本的比例
        recall[i] = tp_cumsum / np.count_nonzero(gnd) # 检测到的正样本占所有正样本的比例
    # Compute the mean precision and recall values over all queries
    # 计算所有查询图像的平均精度和召回率
    mean_precision = np.mean(precision, axis=0)
    mean_recall = np.mean(recall, axis=0)

    # Plot the precision-recall curve
    # plt.plot(mean_recall, mean_precision, 'b-')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Precision-Recall Curve')
    # plt.show()
    return  mean_precision, mean_recall

import time

def get_precision_recall_by_Hamming_Radius(database_output, database_labels, query_output, query_labels, radius=2):
    "投入的标签都需要进行onehot编码 ， 并且是numpy数组"
    # signed_query_output = np.sign(query_output) # -1 0 1 处理
    # signed_database_output = np.sign(database_output)
    signed_query_output = query_output
    signed_database_output = database_output
    bit_n = signed_query_output.shape[1]

    ips = np.dot(signed_query_output, signed_database_output.T)
    ips = (bit_n - ips) / 2

    start_time = time.time()
    ids = np.argsort(ips, 1)
    end_time = time.time()
    sort_time = end_time - start_time
    print("total query: {:d}, sorting time: {:.3f}".format(ips.shape[0], sort_time))

    precX = []
    recX = []
    mAPX = []
    matchX = []
    allX = []
    zero_count = 0
    for i in range(ips.shape[0]):
        if i % 100 == 0:
            tmp_time = time.time()
            end_time = tmp_time
        label = query_labels[i, :]
        label[label == 0] = -1
        idx = np.reshape(np.argwhere(ips[i, :] <= radius), (-1))
        all_num = len(idx)
        if all_num != 0:
            imatch = np.sum(database_labels[idx[:], :] == label, 1) > 0
            match_num = np.sum(imatch)
            precX.append(float(match_num) / all_num)
            matchX.append(match_num)
            allX.append(all_num)
            all_sim_num = np.sum(
                np.sum(database_labels[:, :] == label, 1) > 0)
            recX.append(float(match_num) / all_sim_num)
            if radius < 10:
                ips_trad = np.dot(
                    query_output[i, :], database_output[ids[i, 0:all_num], :].T)
                ids_trad = np.argsort(-ips_trad, axis=0)
                db_labels = database_labels[ids[i, 0:all_num], :]

                rel = match_num
                imatch = np.sum(db_labels[ids_trad, :] == label, 1) > 0
                Lx = np.cumsum(imatch)
                Px = Lx.astype(float) / np.arange(1, all_num + 1, 1)
                if rel != 0:
                    mAPX.append(np.sum(Px * imatch) / rel)
            else:
                mAPX.append(float(match_num) / all_num)

        else:
            print('zero: %d, no return' % zero_count)
            zero_count += 1
            precX.append(float(0.0))
            recX.append(float(0.0))
            mAPX.append(float(0.0))
            matchX.append(0.0)
            allX.append(0.0)
    print("total query: {:d}, sorting time: {:.3f}".format(ips.shape[0], sort_time))
    print("total time: {:.3f}".format(time.time() - start_time))
    return np.mean(np.array(precX)), np.mean(np.array(recX)), np.mean(np.array(mAPX))
def topk_precision(retrieval_hashes, retrieval_labels, query_hashes, query_labels, k):
    "传入一维度标签 ，得到单个数值"
    """
    计算 Top-K Precision

    参数：
    retrieval_hashes：ndarray，shape 为 (N, m)，其中 N 是检索集中图像的数量，m 是哈希码的长度。
    retrieval_labels：ndarray，shape 为 (N,)，其中包含检索集中每个图像的标签。
    query_hashes：ndarray，shape 为 (M, m)，其中 M 是查询集中图像的数量，m 是哈希码的长度。
    query_labels：ndarray，shape 为 (M,)，其中包含查询集中每个图像的标签。
    k：int，计算 Top-K Precision 的 K 值。

    返回：
    topk_precision：float，Top-K Precision 的值。
   """
    topk_precision = 0.0
    for i in range(len(query_hashes)):
        query_hash = query_hashes[i]
        query_label = query_labels[i]
        distances = np.sum(retrieval_hashes != query_hash, axis=1)
        indices = np.argsort(distances)
        topk_labels = retrieval_labels[indices[:k]]
        topk_matches = np.sum(topk_labels == query_label)
        topk_precision += topk_matches / k
    topk_precision /= len(query_hashes)
    return topk_precision

K = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
def topK_precision_list(qB, rB, query_label, retrieval_label, K=K):
    "传入onehot标签"
    num_query = query_label.shape[0]
    p = [0] * len(K)
    for iter in range(num_query):
        gnd = (query_label[iter].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[iter, :], rB).squeeze()
        for i in range(len(K)):
            total = min(K[i], retrieval_label.shape[0])
            ind = torch.sort(hamm)[1][:total]
            # torch.sort()返回两个值第一个是排序后的值的列表和排序后的索引列表只要前K个
            gnd_ = gnd[ind]
            p[i] += gnd_.sum() / total
    p = torch.Tensor(p) / num_query


    # plt.plot(K, p, linestyle="-", marker='D', markevery=0.1 , color='blue' , label = 'DSH')
    # plt.title('CIFAR-10',fontsize=18)
    # plt.grid(True)
    # plt.xlim(1, max(K))
    # plt.ylim(0, 1)
    # plt.xlabel('top-K')
    # plt.ylabel('Precision')
    # # plt.legend()  # 加图例
    # plt.show()
    return p

import numpy as np
from sklearn.preprocessing import normalize
import scipy.io as scio
import torch

def cos(A, B=None):
    """cosine"""
    An = normalize(A, norm='l2', axis=1)
    if (B is None) or (B is A):
        return np.dot(An, An.T)
    Bn = normalize(B, norm='l2', axis=1)
    return np.dot(An, Bn.T)


def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH


def euclidean(A, B=None, sqrt=False):
    aTb = np.dot(A, B.T)
    if (B is None) or (B is A):
        aTa = np.diag(aTb)
        bTb = aTa
    else:
        aTa = np.diag(np.dot(A, A.T))
        bTb = np.diag(np.dot(B, B.T))
    D = aTa[:, np.newaxis] - 2.0 * aTb + bTb[np.newaxis, :]
    if sqrt:
        D = np.sqrt(D)
    return D


def sim_mat(label, label_2=None, sparse=False):
    if label_2 is None:
        label_2 = label
    if sparse:
        S = label[:, np.newaxis] == label_2[np.newaxis, :]
    else:
        S = np.dot(label, label_2.T) > 0
    return S.astype(label.dtype)


def NDCG(qF, rF, qL, rL, k=100):
    # qF: query feature, tensor of size (num_query, code_length)
    # rF: retrieval feature, tensor of size (num_retrieval, code_length)
    # qL: query label, tensor of size (num_query,)
    # rL: retrieval label, tensor of size (num_retrieval,)
    # k: top k retrieval

    num_query = qF.shape[0]
    num_retrieval = rF.shape[0]

    # 计算相似度矩阵
    sim_mat = qF @ rF.T

    # 按相似度排序
    rank_mat = np.argsort(-sim_mat, axis=1)

    # 计算理想的dcg
    ideal_dcg = np.zeros((num_query, k))
    for i in range(num_query):
        pos_idx = np.argwhere(rL==qL[i])[0:k].reshape(-1)
        ideal_dcg[i] = (np.log2(2 + np.arange(k)) * (rL[pos_idx]==qL[i])).sum()

    # 计算dcg
    dcg = np.zeros((num_query, k))
    for i in range(num_query):
        dcg[i] = (np.log2(2 + np.arange(k)) * (rL[rank_mat[i, :k]]==qL[i])).sum()

    # 计算NDCG
    ndcg = dcg / ideal_dcg
    print(f'NDCG@{k}: {ndcg.mean()}')
    return ndcg.mean()


def get_precision_recall_by_Hamming_Radius(database_output, database_labels, query_output, query_labels, radius=2):
    "投入的标签都需要进行onehot编码 ， 并且是numpy数组"
    # signed_query_output = np.sign(query_output)
    # signed_database_output = np.sign(database_output)
    signed_query_output = query_output
    signed_database_output = database_output
    bit_n = signed_query_output.shape[1] # 哈希码长度

    ips = np.dot(signed_query_output, signed_database_output.T) #
    ips = (bit_n - ips) / 2 # 汉明半径

    start_time = time.time()
    ids = np.argsort(ips, 1)
    end_time = time.time()
    sort_time = end_time - start_time
    print("total query: {:d}, sorting time: {:.3f}".format(ips.shape[0], sort_time))

    precX = []
    recX = []
    mAPX = []
    matchX = []
    allX = []
    zero_count = 0
    for i in range(ips.shape[0]):
        if i % 100 == 0:
            tmp_time = time.time()
            end_time = tmp_time
        label = query_labels[i, :]
        label[label == 0] = -1
        idx = np.reshape(np.argwhere(ips[i, :] <= radius), (-1))
        all_num = len(idx)
        if all_num != 0:
            imatch = np.sum(database_labels[idx[:], :] == label, 1) > 0
            match_num = np.sum(imatch)
            precX.append(float(match_num) / all_num)
            matchX.append(match_num)
            allX.append(all_num)
            all_sim_num = np.sum(
                np.sum(database_labels[:, :] == label, 1) > 0)
            recX.append(float(match_num) / all_sim_num)
            if radius < 10:
                ips_trad = np.dot(
                    query_output[i, :], database_output[ids[i, 0:all_num], :].T)
                ids_trad = np.argsort(-ips_trad, axis=0)
                db_labels = database_labels[ids[i, 0:all_num], :]

                rel = match_num
                imatch = np.sum(db_labels[ids_trad, :] == label, 1) > 0
                Lx = np.cumsum(imatch)
                Px = Lx.astype(float) / np.arange(1, all_num + 1, 1)
                if rel != 0:
                    mAPX.append(np.sum(Px * imatch) / rel)
            else:
                mAPX.append(float(match_num) / all_num)

        else:
            print('zero: %d, no return' % zero_count)
            zero_count += 1
            precX.append(float(0.0))
            recX.append(float(0.0))
            mAPX.append(float(0.0))
            matchX.append(0.0)
            allX.append(0.0)
    print("total query: {:d}, sorting time: {:.3f}".format(ips.shape[0], sort_time))
    print("total time: {:.3f}".format(time.time() - start_time))
    return np.mean(np.array(precX)), np.mean(np.array(recX)), np.mean(np.array(mAPX))


