'''
可视化 ： PR 曲线
        TopK-Precison
        NDCG@K
        Phamming2
'''
import torch
import numpy as np
import pandas as pd
import scipy.io as scio
import time
import tqdm
import matplotlib
# matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
import os

def PLOT_PR(Percision ,Recall  , binary_bits , dataset , modelname  , color='blue'):
    dataset = dataset.upper()
    plt.rcParams.update({'font.size': 10})
    plt.title(dataset,fontsize=16)
    plt.plot(Recall, Percision , linestyle="--" , marker = 'D' , markevery=0.1 , color=color,label = 'DSH')
    plt.text(0.5, -0.1, f'(a) PR curve @ {binary_bits}bits', ha='center', va='center', fontsize=16, fontweight='bold', transform=plt.gca().transAxes)
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(np.arange(0 ,1 ,0.1))
    plt.yticks(np.arange(0 ,1 ,0.1))
    # 调整图像的大小
    fig = plt.gcf()
    fig.set_size_inches(9, 9)
    plt.xlabel(f'Recall @ {binary_bits}bits' , fontsize=20)
    plt.ylabel(f'Precision @{binary_bits}' , fontsize=20)
    plt.legend()  # 加图例
    # save the img
    os.makedirs(f'./Saved_PRcruve_img_for_{binary_bits}_{dataset}_{modelname}' , exist_ok=True)
    plt.savefig(f'./Saved_PRcruve_img_for_{binary_bits}_{dataset}_{modelname}/{binary_bits}_{dataset}_{modelname}.png',dpi=500)
    plt.clf()
    # plt.show()

K = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

def PLOT_TOPK_P(TopK , Percision , binary_bits , dataset , modelname , color='blue'):
    dataset = dataset.upper()
    plt.rcParams.update({'font.size': 10})
    plt.title(dataset,fontsize=16)
    plt.plot(TopK, Percision , linestyle="--" , marker = 'D' , markevery=0.1 , color=color ,label = modelname)
    plt.text(0.5, -0.1, f'(b) TopK-Percision @ {binary_bits}bits', ha='center', va='center', fontsize=16, fontweight='bold', transform=plt.gca().transAxes)
    plt.grid(True)
    plt.xlim(0, max(K))
    plt.ylim(0, 1)
    # plt.xticks(np.arange(0 ,1 ,0.1))
    # plt.yticks(np.arange(0 ,1 ,0.1))
    # 调整图像的大小
    fig = plt.gcf()
    fig.set_size_inches(9, 9)
    plt.xlabel('Number of top returned images ' , fontsize=20)
    plt.ylabel(f'Precision @{binary_bits}' , fontsize=20)
    plt.legend()  # 加图例
    # save the img
    os.makedirs(f'./Saved_TopKPercision_img_for_{binary_bits}_{dataset}_{modelname}' , exist_ok=True)
    plt.savefig(f'./Saved_TopKPercision_img_for_{binary_bits}_{dataset}_{modelname}/{binary_bits}_{dataset}_{modelname}.png',dpi=500)
    # plt.show()
    plt.clf()

def PLOT_TOPK_R(TopK , Recall , binary_bits , dataset , modelname , color='blue'):
    dataset = dataset.upper()
    plt.rcParams.update({'font.size': 10})
    plt.title(dataset,fontsize=16)
    plt.plot(TopK, Recall , linestyle="--" , marker = 'D' , markevery=0.1 , color=color ,label = modelname)
    plt.text(0.5, -0.1, f'(b) TopK-Percision @ {binary_bits}bits', ha='center', va='center', fontsize=16, fontweight='bold', transform=plt.gca().transAxes)
    plt.grid(True)
    plt.xlim(0, max(K))
    plt.ylim(0, 1)
    # plt.xticks(np.arange(0 ,1 ,0.1))
    # plt.yticks(np.arange(0 ,1 ,0.1))
    # 调整图像的大小
    fig = plt.gcf()
    fig.set_size_inches(9, 9)
    plt.xlabel('Number of top returned images ' , fontsize=20)
    plt.ylabel(f'Precision @{binary_bits}' , fontsize=20)
    plt.legend()  # 加图例
    # save the img
    os.makedirs(f'./Saved_TopKPercision_img_for_{binary_bits}_{dataset}_{modelname}' , exist_ok=True)
    plt.savefig(f'./Saved_TopKPercision_img_for_{binary_bits}_{dataset}_{modelname}/{binary_bits}_{dataset}_{modelname}.png',dpi=500)
    # plt.show()
    plt.clf()

def PLOT_NDCG(K , NDCG , binary_bits , dataset , modelname , color='blue'):
    dataset = dataset.upper()
    plt.rcParams.update({'font.size': 10})
    plt.title(dataset,fontsize=16)
    plt.plot(K,NDCG , linestyle="--" , marker = 'D' , markevery=0.1 , color=color ,label = modelname)
    plt.text(0.5, -0.1, f'(b) K-NDCG @ {binary_bits}bits', ha='center', va='center', fontsize=16, fontweight='bold', transform=plt.gca().transAxes)
    plt.grid(True)
    plt.xlim(0, max(K))
    plt.ylim(0, 1)
    # plt.xticks(np.arange(0 ,1 ,0.1))
    # plt.yticks(np.arange(0 ,1 ,0.1))
    # 调整图像的大小
    fig = plt.gcf()
    fig.set_size_inches(9, 9)
    plt.xlabel(f'NDCG @ {binary_bits}bits' , fontsize=20)
    plt.ylabel('NDCG' , fontsize=20)
    plt.legend()  # 加图例
    # save the img
    os.makedirs(f'./Saved_K_NDCG_img_for_{binary_bits}_{dataset}_{modelname}' , exist_ok=True)
    plt.savefig(f'./Saved_K_NDCG_img_for_{binary_bits}_{dataset}_{modelname}/{binary_bits}_{dataset}_{modelname}.png',dpi=500)
    # plt.show()
    plt.clf()

def PLOT_PHAMMING2(K , P , binary_bits , dataset , modelname , color='blue'):
    dataset = dataset.upper()
    plt.rcParams.update({'font.size': 10})
    plt.title(dataset,fontsize=16)
    plt.plot(K,P , linestyle="--" , marker = 'D' , markevery=0.1 , color=color ,label = modelname)
    plt.text(0.5, -0.1, f'(b) K-NDCG @ {binary_bits}bits', ha='center', va='center', fontsize=16, fontweight='bold', transform=plt.gca().transAxes)
    plt.grid(True)
    plt.xlim(0, max(K))
    plt.ylim(0, 1)
    # plt.xticks(np.arange(0 ,1 ,0.1))
    # plt.yticks(np.arange(0 ,1 ,0.1))
    # 调整图像的大小
    fig = plt.gcf()
    fig.set_size_inches(9, 9)
    plt.xlabel('Number of bits' , fontsize=20)
    plt.ylabel('P@H <= 2' , fontsize=20)
    plt.legend()  # 加图例
    # save the img
    os.makedirs(f'./Saved_K_PH2_img_for_{binary_bits}_{dataset}_{modelname}' , exist_ok=True)
    plt.savefig(f'./Saved_K_PH2_img_for_{binary_bits}_{dataset}_{modelname}/{binary_bits}_{dataset}_{modelname}.png',dpi=500)
    # plt.show()
    plt.clf()