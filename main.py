import os
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from save_mat import Save_mat
from model import *
from utils import *


def hashing_loss(b, cls, m, alpha):
    """
    compute hashing loss
    automatically consider all n^2 pairs
    """
    y = (cls.unsqueeze(0) != cls.unsqueeze(1)).float().view(-1)
    dist = ((b.unsqueeze(0) - b.unsqueeze(1)) ** 2).sum(dim=2).view(-1)
    loss = (1 - y) / 2 * dist + y / 2 * (m - dist).clamp(min=0)

    loss = loss.mean() + alpha * (b.abs() - 1).abs().sum(dim=1).mean() * 2

    return loss


def train(epoch, dataloader, net, optimizer, m, alpha):
    '''
    在这段代码中，2 * opt.binary_bi
    ts是作为m参数传递给hashing_loss函数的。这是因为在深度学习哈希模型中，哈希码的长度通常是二进制位数的两倍。
    例如，如果我们希望生成32位哈希码，则实际使用的哈希码长度应为64位。

    这是因为在深度学习哈希模型中，哈希码是通过将原始特征向量投射到一个低维二进制空间中得到的。如果哈希码长度太短，
    可能无法充分表示图像的语义信息，从而导致检索准确性下降。因此，通常使用较长的哈希码来提高检索准确性。

    在这个代码中，2 * opt.binary_bits是为了确保使用的哈希码长度足够长，以充分表示图像的语义信息。
    具体来说，opt.binary_bits是作为命令行参数传递给代码的，表示希望生成的哈希码长度。通过将其乘以2，
    可以得到实际使用的哈希码长度。
    '''
    accum_loss = 0
    net.train()
    for i, (img, cls) in enumerate(dataloader):
        img, cls = [x.cuda() for x in (img, cls)]

        optimizer.zero_grad()
        b = net(img)
        loss = hashing_loss(b, cls, m, alpha)

        loss.backward()
        optimizer.step()
        accum_loss += loss.detach().item()

        print(f'[{epoch}][{i}/{len(dataloader)}] loss: {loss.item():.4f}')
    return accum_loss / len(dataloader)


def test(epoch, dataloader, net, m, alpha):
    accum_loss = 0
    net.eval()
    with torch.no_grad():
        for img, cls in dataloader:
            img, cls = [x.cuda() for x in (img, cls)]

            b = net(img)
            loss = hashing_loss(b, cls, m, alpha)
            accum_loss += loss.detach().item()

    accum_loss /= len(dataloader)
    print(f'[{epoch}] val loss: {accum_loss:.4f}')
    return accum_loss


def main():
    parser = argparse.ArgumentParser(description='train DSH')
    parser.add_argument('--cifar', default='../dataset/cifar', help='path to cifar')
    parser.add_argument('--weights', default='', help="path to weight (to continue training)")
    parser.add_argument('--outf', default='checkpoints', help='folder to output model checkpoints')
    parser.add_argument('--checkpoint', type=int, default=50, help='checkpointing after batches')

    parser.add_argument('--batchSize', type=int, default=512, help='input batch size')
    parser.add_argument('--ngpu', type=int, default=0, help='which GPU to use')

    parser.add_argument('--binary_bits', type=int, default=128, help='length of hashing binary')
    parser.add_argument('--alpha', type=float, default=0.01, help='weighting of regularizer')

    parser.add_argument('--niter', type=int, default=501, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

    opt = parser.parse_args()
    print(opt)

    os.makedirs(opt.outf, exist_ok=True)
    choose_gpu(opt.ngpu)
    feed_random_seed()
    train_loader, test_loader = init_cifar_dataloader(opt.cifar, opt.batchSize)
    logger = SummaryWriter()

    # setup net
    net = DSH(opt.binary_bits)
    resume_epoch = 0
    print(net)
    if opt.weights:
        print(f'loading weight form {opt.weights}')
        resume_epoch = int(os.path.basename(opt.weights)[:-4])
        net.load_state_dict(torch.load(opt.weights, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

    net.cuda()

    # setup optimizer
    optimizer = optim.Adam(net.parameters(), lr=opt.lr, weight_decay=0.004)

    for epoch in range(resume_epoch, opt.niter):
        train_loss = train(epoch, train_loader, net, optimizer, 2 * opt.binary_bits, opt.alpha)
        logger.add_scalar('train_loss', train_loss, epoch)

        test_loss = test(epoch, test_loader, net, 2 * opt.binary_bits, opt.alpha)
        logger.add_scalar('test_loss', test_loss, epoch)

        if epoch % opt.checkpoint == 0:
            # compute mAP by searching testset images from trainset
            '''
            这里保存的二进制哈希码和标签是整个数据集的表现。具体来说，compute_result函数计算给定数据集的每个数据样本的二进制哈希码和标签，并将它们保存在两个张量bs和clses中。在计算mAP时，它使用这两个张量来计算训练集和测试集之间的相似度得分，并根据得分排序，从而获得检索结果。
            在保存表现最好的哈希码和标签时，它使用compute_result函数计算训练集和测试集的哈希码和标签，并将它们保存在变量trn_binary、trn_label、tst_binary和tst_label中。这些哈希码和标签可以用于计算mAP和进行图像检索任务。
            需要注意的是，在这段代码中，compute_result函数计算的哈希码和标签是针对整个数据集的，而不是特定的epoch。因此，它计算的是整个数据集的表现，而不是单个epoch的表现。如果您想要保存特定epoch的哈希码和标签，您需要在train或test函数中进行修改，以便在特定epoch结束时保存哈希码和标签，而不是在计算mAP时保存它们。
            '''
            trn_binary, trn_label = compute_result(train_loader, net)
            tst_binary, tst_label = compute_result(test_loader, net)
            mAP = compute_mAP(trn_binary, tst_binary, trn_label, tst_label)
            print(f'[{epoch}] retrieval mAP: {mAP:.4f}')
            logger.add_scalar('retrieval_mAP', mAP, epoch)
            # 保存哈希码和标签
            Save_mat(epoch=epoch,output_dim=opt.binary_bits,datasets="cifar10",
                     query_labels=tst_label , retrieval_labels=trn_label ,
                     query_img=tst_binary , retrieval_img=trn_binary)
            # save checkpoints
            torch.save(net.state_dict(), os.path.join(opt.outf, f'{epoch:03d}.pth'))


if __name__ == '__main__':
    main()
