# DSH_DeepSupervisedHashing
A new version for https://github.com/weixu000/DSH-pytorch
* 2022-5-21更新 object文件夹可以轻松帮你评估6个指标（mAP,PRcurve,NDCG@1000,TopK-Precision,TopK-Recall,P@H<=2)
* 2022-5-21updated dictory named object will help you easily predict six indexes （mAP,PRcurve,NDCG@1000,TopK-Precision,TopK-Recall,P@H<=2) 

# Instruction(说明)
* PR——cruves是一个计算PR曲线指标的模块 ， save_mat是一个把模型生成的哈希码和标签保存为.mat文件的模块
# where you can find the parper
-PyTorch implementation of paper [Deep Supervised Hashing for Fast Image Retrieval](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_Deep_Supervised_Hashing_CVPR_2016_paper.pdf)
# It is a brandnew version of DSH_DeepSupervisedHashing 
* Which is rewrited by GPT3.5 (I also participated in this job , Because some errors will happen to AI)😆
* 😢 ， but I also modified the parameters in the source code, and a pooling layer in the network ( which can perform a little better than it used to be)
# tips you might interested in 
* Adding a Savemat function in main.py so that you can use it to Visualize some metrics （ like PrCruve some what .....💐)
* provide a useful method to Visualize Prcruve , U can use Dataspell to do it by using the csv file
##How to Run
* I intended to make it easy for reading. You can easily run it by
```shell
python main.py
```
*commandline uasge have a nice-looking help
```shell
pyhon main.py -h
```
## packages
pytorch == 1.12.0 + cu113

torchvision == 0.13.0 + cu113

tqdm

scipy

tornado

matplotlib
