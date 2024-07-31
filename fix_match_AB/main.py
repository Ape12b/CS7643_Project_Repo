import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import get_cifar10, get_cosine_schedule_with_warmup
from test_train import train_fixmatch
import torch.backends.cudnn as cudnn
from get_wide_resnet import create_wideresnet


def main():
    args = {
        'gpu_id': 0,
        'num_workers': 2,
        'num_labeled': 400,
        'epochs': 20000,
        'batch_size': 64,
        'lr': 0.01,
        'warmup': 0,
        'wdecay': 5e-4,
        'nesterov': True,
        'use_ema': True,
        'ema_decay': 0.999,
        'mu': 5,
        'lambda_u': 1,
        'threshold': 0.95,
        'k_img': 32768,
        'out': 'result',
        'resume': '',
        'seed': -1,
        'amp': False,
        'opt_level': 'O1'
    }


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_labeled_dataset, train_unlabeled_dataset, test_dataset = get_cifar10('./data', args['num_labeled'], 
                                                                               args['k_img'], args['k_img'] * args['mu'])

    labeled_loader = DataLoader(train_labeled_dataset, batch_size=args['batch_size'], 
                                shuffle=True, num_workers=args['num_workers'], drop_last=True)
    unlabeled_loader = DataLoader(train_unlabeled_dataset, batch_size=args['batch_size'] * args['mu'], 
                                  shuffle=True, num_workers=args['num_workers'], drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], 
                             shuffle=False, num_workers=args['num_workers'])
    
    
    model = create_wideresnet(num_blocks=28, width_factor=2)
    model = model.to(device)  # Move model to the device
    criterion = nn.CrossEntropyLoss().cuda()
    
    wd_params, non_wd_params = [], []
    for param in model.parameters():
        if len(param.size()) == 1:
            non_wd_params.append(param)
        else:
            wd_params.append(param)
    param_list = [
        {'params': wd_params}, {'params': non_wd_params, 'weight_decay': 0}]
    optimizer = torch.optim.SGD(param_list, lr=args['lr'], weight_decay=args['wdecay'],
        momentum=0.9, nesterov=True)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args['warmup'], args['epochs'])

    model = train_fixmatch(model, labeled_loader, unlabeled_loader, test_loader, 
                           criterion, optimizer, scheduler, device, 
                           num_epochs=args['epochs'], threshold=args['threshold'], lu_weight=args['lambda_u'])

if __name__ == '__main__':
    cudnn.benchmark = True
    main()