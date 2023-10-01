'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models
import torchvision.models
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import PIL 




from robustbench.data import load_cifar100c 
from robustbench.utils import load_model
from robustbench.utils import clean_accuracy 

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name])) 
model_names.append('resnet50')

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--data_dir', default='data/cifar100_original_data', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--data_expanded_dir', default='data/cifar100_original_data', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--level', default=5, type=int, metavar='N')
# Optimization options
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[41, 81],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')  
# Checkpoints
parser.add_argument('--pretrained',  default=False, action='store_true')
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--accumulate',  type=int, default= 0)
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

#需要在一样的image size
def corrpution_evaluate(model, logger):   
    transform_test = transforms.Compose([
        transforms.Resize(224, interpolation=PIL.Image.BICUBIC),    
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    tesize = 10000
    common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
	 				'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
	 				'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    for corruption_type in common_corruptions: 
        teset_raw = np.load('./data//CIFAR-100-C/%s.npy' %(corruption_type))
        teset_raw = teset_raw[(args.level-1)*tesize: args.level*tesize]
        teset = torchvision.datasets.CIFAR100( "./data/", train=False, download=True, transform=transform_test)
        teset.data = teset_raw  
        teloader = torch.utils.data.DataLoader(teset, batch_size=100, shuffle=False, num_workers=args.workers)
                        
        #acc = clean_accuracy(model, x_test, y_test, 100) 
        top1 = AverageMeter()
        with torch.no_grad():
            for batch_idx, (x_test, y_test) in enumerate(teloader): 
                 
                x_test, y_test = x_test.cuda(), y_test.cuda() 
                outputs = model(x_test)
                prec1, prec5 = accuracy(outputs.data, y_test.data, topk=(1, 5))
                top1.update(prec1.item(), x_test.size(0))
            
        print("Corruption type {}, severity {}: acc {}".format(corruption_type, args.level, top1.avg))
        #logger.info(f"error % [{corruption_type}{args.level}]: {acc:.2%}")



def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
         os.makedirs(args.checkpoint)

    print('==> Preparing dataset %s' % args.dataset) 
    transform_train = transforms.Compose([
        transforms.Resize((224,224), interpolation=PIL.Image.BICUBIC),
        transforms.RandomCrop(224),
        transforms.RandomRotation(15,),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
 
    
    transform_test = transforms.Compose([
        transforms.Resize(224, interpolation=PIL.Image.BICUBIC),   
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])


    

    # NORM =((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # te_transforms = transforms.Compose([transforms.ToTensor(),
    #                                     transforms.Normalize(*NORM)])

    # common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
	# 				'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
	# 				'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']



    # def prepare_test_data(args):
    #     if args.dataset == 'cifar100':
    #         tesize = 10000 
    #         elif args.corruption in common_corruptions:
    #             print('Test on %s level %d' %(args.corruption, args.level))
    #             teset_raw = np.load('./data/CIFAR-100-C/%s.npy' %(args.corruption))
    #             teset_raw = teset_raw[(args.level-1)*tesize: args.level*tesize]
    #             teset = torchvision.datasets.CIFAR100(root="./data",  train=False, download=True, transform=te_transforms)
    #             teset.data = teset_raw 
    #         else:
    #             raise Exception('Corruption not found!')
    #     else:
    #         raise Exception('Dataset not found!')

    #     if not hasattr(args, 'workers'):
    #         args.workers = 1
    #     teloader = torch.utils.data.DataLoader(teset, batch_size=args.batch_size,
    #                                             shuffle=True, num_workers=args.workers)
    #     return teset, teloader 

    # Data
    
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100


    

    original_trainset = datasets.ImageFolder(args.data_dir,  transform_train) 
    expanded_trainset = datasets.ImageFolder(args.data_expanded_dir,  transform_train) 
    new_trainset = torch.utils.data.ConcatDataset([original_trainset, expanded_trainset])
    #trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    print("{} Mode: Contain {} images".format("Train", len(new_trainset)))
    trainloader = data.DataLoader(new_trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    #testset = datasets.ImageFolder(args.data_dir+"/test",  transform_test) 
    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    
    # Model
    print("==> creating model '{}'".format(args.arch))
    # create model
    #num_classes = 1000
    dim_feature = 2048    
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    baseWidth=args.base_width,
                    cardinality=args.cardinality,
                )
    elif args.arch == 'resnet50':
        model = torchvision.models.resnet50(pretrained=args.pretrained) 
        model.fc = nn.Linear(dim_feature, num_classes)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
        
        
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Resume
    title = 'cifar-100-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


    if args.evaluate:
        print('\nEvaluation only')
        model.eval()
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        #adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        if is_best:
             logger.write("The best performance:" + str(best_acc))          
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best,best_acc, checkpoint=args.checkpoint)
        scheduler.step()
    corrpution_evaluate(model, logger)
    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)

def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step 
        if args.accumulate==0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else: 
            accumulate_step =args.accumulate
            loss = loss/accumulate_step
            loss.backward()
            if ((batch_idx+1)%accumulate_step)==0:
                optimizer.step()
                optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    torch.cuda.empty_cache() 
    return (losses.avg, top1.avg)

def test(val_loader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(val_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    torch.cuda.empty_cache() 
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, best_acc, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        print("The best performance:", best_acc)
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))



def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()

