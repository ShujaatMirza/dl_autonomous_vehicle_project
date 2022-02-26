import argparse
import os, sys, numpy as np
from time import time
import logging
import yaml

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from jigsaw_model import Network
from data_helper import UnlabeledDataset
from model import get_transform_jigsaw


def train(image_folder, batch_size=2, max_training_scene=25, epochs=0, learning_rate=1e-3, classes=100,
        cores=0, checkpoint_dir='checkpts_jigsaw', model_dir='./model', verbose=False,iter_start=0, device=None):
    
    if device is not None:
        logger.info(('Using GPU %d'%device))
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=str(device)
    else:
        logger.info('CPU mode')
    
    logger.info('Process number: %d'%(os.getpid()))
    
    transform = transforms.ToTensor()

    unlabeled_scene_index_train = np.arange(106)[:max_training_scene]
    unlabeled_trainset = UnlabeledDataset(image_folder=image_folder,
                                    scene_index=unlabeled_scene_index_train,
                                    first_dim='image',
                                    transform=get_transform_jigsaw
                                    )
    train_loader = torch.utils.data.DataLoader(unlabeled_trainset, batch_size=batch_size, shuffle=True, num_workers=cores)

    iter_per_epoch = len(unlabeled_trainset)/batch_size
    logger.info('Images: train %d'%(len(unlabeled_trainset)))
    
    # Network initialize
    net = Network(classes)
    if device is not None:
        net.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),lr=learning_rate,momentum=0.9,weight_decay = 5e-4)
    
    # Train the Model
    logger.info(('Start training: lr %f, batch size %d, classes %d'%(learning_rate,batch_size,classes)))
    logger.info(('Checkpoint: '+ checkpoint_dir))
    batch_time, net_time = [], []
    steps = iter_start
    for epoch in range(int(iter_start/iter_per_epoch),epochs):
        if epoch%10==0 and epoch>0:
            #test(net,criterion,logger,val_loader,steps)
            test(image_folder, net,args.log_level,steps,device)
        lr = adjust_learning_rate(optimizer, epoch, init_lr=learning_rate, step=20, decay=0.1)
        end = time()
        for i, (images, labels) in enumerate(train_loader):
            batch_time.append(time()-end)
            if len(batch_time)>100:
                del batch_time[0]
            
            images = Variable(images[0])
            labels = Variable(labels)
            if device is not None:
                images = images.cuda()
                labels = labels.cuda()
            
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            t = time()
            outputs = net(images)
            net_time.append(time()-t)
            if len(net_time)>100:
                del net_time[0]
            
            prec1, prec5 = compute_accuracy(outputs.cpu().data, labels.cpu().data, topk=(1, 5))
            acc = prec1

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss = float(loss.cpu().data.numpy())

            if steps%20==0:
                logger.info(('[%2d/%2d] %5d) [batch load % 2.3fsec, net %1.2fsec], LR %.5f, Loss: % 1.3f, Accuracy % 2.2f%%' %(
                            epoch+1, epochs, steps, 
                            np.mean(batch_time), np.mean(net_time),
                            lr, loss,acc)))
   
            steps += 1
            end = time()

        #Write to checkpoint at the end of epoch
        filename = '%s/100_jps_%03i.pth'%(checkpoint_dir,epoch)
        net.save(filename)
        logger.info('Saved: '+ checkpoint_dir)

def test(image_folder, net,logger,steps,device,classes,weights):
    
    unlabeled_scene_index_valid = np.arange(1)
    unlabeled_testset = UnlabeledDataset(image_folder=image_folder,
                                scene_index=unlabeled_scene_index_valid,
                                first_dim='image',
                                transform=get_transform_jigsaw
                                )
    val_loader = torch.utils.data.DataLoader(unlabeled_testset, batch_size=1, shuffle=True, num_workers=1)
    
    logger.info('Evaluating network.......')
    logger.info('Images: validation %d'%(len(unlabeled_testset)))

    net = Network(classes)
    if device is not None:
        net.cuda()
    
    accuracy = []
    net.eval()
    for i, (images, labels) in enumerate(val_loader):
        images = Variable(images[0])
        if device is not None:
            images = images.cuda()

        # Forward + Backward + Optimize
        outputs = net(images)
        outputs = outputs.cpu().data

        prec1, prec20 = compute_accuracy(outputs, labels, topk=(1,20))
        accuracy.append(prec20)
        logger.info('Accuracy top 20:  %.2f%%' %(prec20))

    logger.info('Mean Testing Accuracy: %.2f%%' %(np.mean(accuracy)))


def eval(image, net,logger,steps,device,classes,weights):
    logger.info('Pretext training')
    
    net = Network(classes)
    if device is not None:
        net.cuda()
    
    net.load(weights)
    net.eval()  

    images = Variable(images)
    if device is not None:
        images = images.cuda()

    # Forward + Backward + Optimize
    outputs = net(images)
    outputs = outputs.cpu().data

def adjust_learning_rate(optimizer, epoch, init_lr=0.1, step=30, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (decay ** (epoch // step))
    print('Learning Rate %f'%lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def compute_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Self-Supervised Pretext task: JigsawPuzzleSolver on unlabeled data')
    parser.add_argument('-mode', '--mode', choices=['train', 'test'], help='integer argument')
    parser.add_argument('-config', '--config', type=str, required=True, help='string argument')
    parser.add_argument('--log_level', '--log_level', type=int, choices=[0, 10, 20, 30, 40, 50], default=20, help='integer argument')
    parser.add_argument('--verbose', '--verbose', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger(__name__)
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)
    
    # Load a configuration file
    with open(args.config, 'r') as yml:
        config = yaml.safe_load(yml)

    image_folder = config['data']['image_folder']
    
    if args.mode == 'train':
        # Sample command: `python jigsaw_train.py -config selfsup.config -mode train -log_level 10`
        train(image_folder,
                batch_size=config['train']['batch_size'],
                max_training_scene=config['train']['max_training_scene'],
                epochs=config['train']['epochs'],
                learning_rate=float(config['param']['learning_rate']),
                classes=config['param']['classes'],
                cores=config['param']['num_cores'],
                checkpoint_dir=config['train']['checkpoint'],
                model_dir=config['data']['model_dir'],
                verbose=args.verbose,
                iter_start=config['param']['iter_start'],
                device=config['param']['gpu_number'])
    else:
        # Sample command: (Needs to be checked as modified) `python jigsaw_train.py -config selfsup.config -mode test -checkpoint_path model/model_at_epoch_3.pt -log_level 10 \
        test(image_folder,
            net=None,
            logger=logger,
            steps=0,
            device=config['param']['gpu_number'],
            classes=config['param']['classes'],
            weights=config['test']['weights'])

"""Code in funtion get_transform_jigsaw has been adopted from Biagio Brattoli's code"""