import time
from functools import wraps
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms




def init_cifar_dataloader(root, batchSize):
    """load dataset"""
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    trainset = CIFAR10(root, train=True, download=True, transform=transform_train)
    train_loader = DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=4, pin_memory=True)
    print(f'train set: {len(train_loader.dataset)}')

    testset = CIFAR10(root, train=False, download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=batchSize * 4, shuffle=False, num_workers=4, pin_memory=True)
    print(f'val set: {len(test_loader.dataset)}')

    return train_loader, test_loader


def timing(f):
    """print time used for function f"""

    @wraps(f)
    def wrapper(*args, **kwargs):
        time_start = time.time()
        ret = f(*args, **kwargs)
        print(f'total time = {time.time() - time_start:.4f}')
        return ret

    return wrapper


def compute_result(dataloader, net):
    bs, clses = [], []
    net.eval()
    with torch.no_grad():
        for img, cls in dataloader:
            clses.append(cls)
            bs.append(net(img.cuda()).data.cpu())
    return torch.sign(torch.cat(bs)), torch.cat(clses)


@timing
def compute_mAP(trn_binary, tst_binary, trn_label, tst_label):
    """
    compute mAP by searching testset from trainset
    https://github.com/flyingpot/pytorch_deephash
    """
    for x in trn_binary, tst_binary, trn_label, tst_label: x.long()

    AP = []
    Ns = torch.arange(1, trn_binary.size(0) + 1)
    for i in range(tst_binary.size(0)):
        query_label, query_binary = tst_label[i], tst_binary[i]
        _, query_result = torch.sum((query_binary != trn_binary).long(), dim=1).sort()
        correct = (query_label == trn_label[query_result]).float()
        P = torch.cumsum(correct, dim=0) / Ns
        AP.append(torch.sum(P * correct) / torch.sum(correct))
    mAP = torch.mean(torch.Tensor(AP))
    return mAP


def choose_gpu(i_gpu):
    """choose current CUDA device"""
    torch.cuda.set_device(i_gpu)
    cudnn.benchmark = True


def feed_random_seed(seed=np.random.randint(1, 10000)):
    """feed random seed"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
