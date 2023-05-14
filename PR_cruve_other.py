import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import os




def plot_pr_curve(query_binary, retrieval_binary, query_label, retrieval_label):
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

    Returns:
        None
    """

    # Convert the labels to int32 type to avoid indexing errors
    ## 将标签转换为 int32 类型，以避免索引错误
    query_label = query_label.astype(np.int32)
    retrieval_label = retrieval_label.astype(np.int32)

    # Calculate the Hamming distances between query and retrieval binary codes
    # 计算查询图像和检索图像之间的汉明距离
    query_binary[query_binary == -1] = 0
    retrieval_binary[retrieval_binary == -1] = 0

    hamming_dist = np.count_nonzero(query_binary[:, np.newaxis, :]
                                    != retrieval_binary[np.newaxis, :, :], axis=2)

    # Sort the retrieval samples by ascending order of Hamming distance
    # 将检索图像按汉明距离排序
    idx = np.argsort(hamming_dist, axis=1)

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
        print(gnd)
        # Compute the cumulative sums of true positives and false positives
        # 计算真正例和假正例的累积和
        tp_cumsum = np.cumsum(gnd)
        fp_cumsum = np.cumsum(~gnd)

        # Compute the precision and recall values
        # 计算精度和召回率
        precision[i] = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall[i] = tp_cumsum / np.count_nonzero(gnd)

    # Compute the mean precision and recall values over all queries
    # 计算所有查询图像的平均精度和召回率
    mean_precision = np.mean(precision, axis=0)
    mean_recall = np.mean(recall, axis=0)
    #
    # # Plot the precision-recall curve
    # # plt.grid(True,which='both', ls=':', color='gray' , alpha=0.5)
    # # plt.rcParams['figure.dpi'] = 2400
    # plt.xticks(np.arange(0 , 1 , 0.1))
    # plt.yticks(np.arange(0 , 1 , 0.1))
    # # plt.plot(mean_recall, mean_precision ,  linestyle='-',marker='.' , color='blue' , label='DSH')
    # # plt.xlabel('Recall')
    # # plt.ylabel('Precision')
    # # plt.title('Precision-Recall Curve')
    # plt.xlim(0,1.0)
    # plt.ylim(0,1.0)
    # # plt.show()
    # plt.plot(mean_recall, mean_precision, linestyle="-", marker='D', color='blue', label='DSH')
    # plt.text(0.5, -0.1, '(a) PR curve @ 16bits', ha='center', va='center', fontsize = 16, fontweight = 'bold', transform = plt.gca().transAxes)
    # plt.grid(True)
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # # 调整图像的大小
    # fig = plt.gcf()
    # fig.set_size_inches(4, 4)
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.legend()  # 加图例
    # plt.show()
    return mean_recall , mean_precision

data = scio.loadmat('../DSH128bits_cifar_Hashcode/128-500-cifar10-DSH.mat')
qB = np.array(data['q_img'])
rB = np.array(data['r_img'])
q_l = np.array(data['q_l'])
r_l = np.array(data['r_l'])
print(qB.shape, r_l.shape)
q_l = np.squeeze(q_l)
r_l = np.squeeze(r_l)
print(q_l.shape)
R , P = plot_pr_curve(query_label=q_l, retrieval_label=r_l, query_binary=rB, retrieval_binary=qB)
os.makedirs('128bits_cifar_csv' , exist_ok=True)
np.savetxt('./128bits_cifar_csv/Recall.csv' , R )
np.savetxt('./128bits_cifar_csv/Percision.csv', P)