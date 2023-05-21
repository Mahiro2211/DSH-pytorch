from DSH_load import *
import argparse
def main():
    print('mat文件的命名规则是{binary_bits}-{dataset}-{modelname}')
    parser = argparse.ArgumentParser(description='Get DSH index')
    parser.add_argument('--file',default='../cifar10_mat',help='path to mat')
    parser.add_argument('--dataset',default='cifar10',help='name of dataset')
    parser.add_argument('--modelname',default='DSH',help='modelname')
    parser.add_argument('--label',default=10,help='onehot')
    tp = parser.parse_args()
    print(tp)
    get_index = DSH_index(filepath=tp.file,dataset=tp.dataset,modelname=tp.modelname)
    get_index.topK_recall()
    # get_index.prcurve(is_img=True)
    # get_index.topK_precision(num=tp.label)
    # get_index.phamming2(num=tp.label)
    # get_index.NDCG_1000()
if __name__ == '__main__':
    main()

