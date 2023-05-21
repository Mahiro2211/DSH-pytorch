import numpy as np
import torch

from utils import *
from visualize_fn import *
import scipy.io as scio
import glob

class DSH_index():
    def __init__(self , dataset , filepath ,color='blue' , modelname='DSH' ,K=[1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]):
        print('mat文件的命名规则是{binary_bits}-{dataset}-{modelname}')
        self.modelname = modelname
        self.dataset = dataset
        self.filepath = filepath
        self.color = color
        self.bits = 16
        self.K = K

    def load_mat(self): # 迭代器
        file_list = glob.glob(os.path.join(self.filepath,f'*{self.dataset}*.mat'))
        print(file_list)
        file_basename = [os.path.splitext(os.path.basename(f))[0] for f in file_list]
        mat_dict = {i : scio.loadmat(file_list[i]) for i in range(len(file_list))}
        for i in range(len(file_list)):
            qb , ql = mat_dict[i]['q_img'] , mat_dict[i]['q_l']
            rb , rl = mat_dict[i]['r_img'] , mat_dict[i]['r_l']
            yield qb,rb,ql,rl

    def prcurve(self,is_img=False):
        iter_mat = iter(self.load_mat())
        num_pic = len(list(enumerate(iter_mat)))
        iter_mat = iter(self.load_mat())
        while True :
            try:
                qb , rb , ql , rl = next(iter_mat)
                self.bits = qb.shape[1]
                if is_img :
                    ql = np.squeeze(ql)
                    rl = np.squeeze(rl)
                    P,R = PR_curve_focus_on_retrieval_img(query_binary=qb,query_label=ql,retrieval_binary=qb,retrieval_label=ql)
                    save_csv('saved_index_pr',P,R,f'{self.bits}_{self.dataset}_Precison',f'{self.bits}_{self.dataset}_Recall')
                    # PLOT_PR(Percision=P , Recall=R ,binary_bits=self.bits ,dataset=self.dataset , color=self.color,modelname=self.modelname)
                else :
                    qb , rb= torch.from_numpy(qb) , torch.from_numpy(rb)
                    ql = one_hot(ql,num=10,np_type=False)
                    rl = one_hot(rl,num=10,np_type=False)
                    P ,R = PR_curve_focus_on_binary_length(qB=qb,query_label=ql,rB=rb,retrieval_label=rl)
                    save_csv('saved_index_pr',P,R,f'{self.bits}_{self.dataset}_Precison',f'{self.bits}_{self.dataset}_Recall')
                    # PLOT_PR(Percision=P , Recall=R ,binary_bits=self.bits ,dataset=self.dataset , color=self.color,modelname=self.modelname)
            except StopIteration :
                break
    def topK_recall(self):
        iter_mat = iter(self.load_mat())
        num_pic = len(list(enumerate(iter_mat)))
        iter_mat = iter(self.load_mat())
        while True:
            try:
                qb, rb, ql, rl = next(iter_mat)
                self.bits = qb.shape[1]
                ql = np.squeeze(ql)
                rl = np.squeeze(rl)
                recallK = []
                for i in self.K:
                    print(i)
                    a = topk_recall(qb,rb,ql,rl,i)
                    print(f'{self.bits}_top{i}_recall is {a}')
                    recallK.append(a)
                save_csv('saved_TopK_recall', self.K, recallK, 'K', f'{self.bits}_{self.dataset}_NDCG')
                # PLOT_NDCG(self.K,ndcg,binary_bits=self.bits,dataset=self.dataset,modelname=self.modelname)
            except StopIteration:
                break

    def topK_precision(self,num=10):
        iter_mat = iter(self.load_mat())
        num_pic = len(list(enumerate(iter_mat)))
        iter_mat = iter(self.load_mat())
        while True:
            try:
                qb , rb , ql , rl = next(iter_mat)
                self.bits = qb.shape[1]
                qb = torch.from_numpy(qb)
                rb = torch.from_numpy(rb)
                ql = one_hot(ql,num=num)
                rl = one_hot(rl,num=num)
                P = topK_precision_list(qB=qb ,query_label=ql,rB=rb,retrieval_label=rl)
                save_csv('saved_topK_precision' , K , P , 'K' , f'{self.bits}_{self.dataset}_Precision')
                # PLOT_TOPK_P(self.K,P,binary_bits=self.bits,dataset=self.dataset,modelname=self.modelname,color=self.color)
            except StopIteration:
                break
    def NDCG_1000(self):
        "numpy_array"
        iter_mat = iter(self.load_mat())
        num_pic = len(list(enumerate(iter_mat)))
        iter_mat = iter(self.load_mat())
        while True:
            try:
                qb , rb , ql , rl = next(iter_mat)
                self.bits = qb.shape[1]
                ql = np.squeeze(ql)
                rl = np.squeeze(rl)
                ndcg = []
                for i in self.K :
                    print(i)
                    a = NDCG(qb,rb,ql ,rl,k=i)
                    ndcg.append(a)
                save_csv('saved_K_NDCG' , self.K , ndcg , 'K' , f'{self.bits}_{self.dataset}_NDCG')
                # PLOT_NDCG(self.K,ndcg,binary_bits=self.bits,dataset=self.dataset,modelname=self.modelname)
            except StopIteration:
                break

    def phamming2(self , num=10):
        iter_mat = iter(self.load_mat())
        num_pic = len(list(enumerate(iter_mat)))
        iter_mat = iter(self.load_mat())
        while True:
            try:
                qb , rb , ql , rl = next(iter_mat)
                self.bits = qb.shape[1]
                if self.bits != 64 :
                    continue
                ql = one_hot(ql ,num = num , np_type=True)
                rl = one_hot(rl, num = num ,np_type=True)
                hash_length = [i for i in range(1 , self.bits + 1)]
                ph2 = []
                recall_ph2 =[]
                Map = []
                for i in hash_length :
                    print(i)
                    input_qb = qb[:,:i]
                    input_rb = rb[:,:i]
                    precision , recall , map = get_precision_recall_by_Hamming_Radius(database_output=input_rb,database_labels=rl,
                                                                                      query_output=input_qb,query_labels=ql)
                    recall_ph2.append(recall)
                    Map.append(map)
                    ph2.append(precision)
                save_csv('saved_recallph2_map' , recall_ph2 , Map , f'{self.bits}_{self.dataset}_recall_PH2',f'{self.bits}_{self.dataset}_map')
                save_csv('saved_hash_ph2' , hash_length , ph2 , 'K' , f'{self.bits}_{self.dataset}_ph2')
                # PLOT_NDCG(hash_length,ph2,binary_bits=self.bits,dataset=self.dataset,modelname=self.modelname)
            except StopIteration:
                break






#%%

#%%
