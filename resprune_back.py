import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import os
import argparse
import numpy as np
from collections import OrderedDict

import config
import models

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--percent', type=float, default=0.2,
                        help='scale sparse rate (default: 0.2)')
    parser.add_argument('--save_path', default='./exp/channel_prune', type=str, metavar='PATH',
                        help='path to save dir')
    parser.add_argument('--prune', default='./exp/pretrained/model_best.pth.tar', type=str, metavar='PATH',
                        help='path to pretrained model')
    return parser.parse_args()

# initial
channel_pruning = True
args = get_args()
use_cuda = config.use_gpu and torch.cuda.is_available()
torch.manual_seed(config.seed)
if use_cuda:
    torch.cuda.manual_seed(config.seed)
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
np.set_printoptions(precision=3, suppress=True)

# dataset
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
if config.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(config.data_dir, train=True, download=True,
                         transform=transforms.Compose([
                             transforms.Pad(4),
                             transforms.RandomCrop(32),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                         ])),
        batch_size=config.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(config.data_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=config.test_batch_size, shuffle=True, **kwargs)
else:
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(config.data_dir, train=True, download=True,
                          transform=transforms.Compose([
                              transforms.Pad(4),
                              transforms.RandomCrop(32),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                          ])),
        batch_size=config.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(config.data_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=config.test_batch_size, shuffle=True, **kwargs)

# model
checkpoint = torch.load(args.prune)
model = models.__dict__[config.network](depth=config.depth, cfg=None, num_classes=config.num_classes)
best_prec1 = checkpoint['best_prec1']
model.load_state_dict(checkpoint['state_dict'])
print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.prune, checkpoint['epoch'], best_prec1))
if use_cuda:
    model.cuda()
print(model)

# prune
total = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        total += m.weight.data.shape[0]
print(total)

#for layer, module in enumerate(model.modules()):
#    print(layer)
#    print(module)

def get_pt():
    prune_table = [[] for _ in range(len(list(model.modules())))]
    not_prune = []
    n = (config.depth - 2) // 9
    count = -1
    for layer, module in enumerate(model.modules()):
        if isinstance(module, models.resnet_v2.Bottleneck):
            count += 1
            if count % n != 0:
                continue
            cap = layer + 6
            for i in range(n-1):
                prune_table[cap].append(layer + 10 + i*8 + 6)
                not_prune.append(layer + 10 + i*8 + 6)
    return prune_table, not_prune
pruneTable,notPrune = get_pt()
#print(pruneTable)
#print(notPrune)

class Pruner:
    def __init__(self, model, num_pruned):
        self.model = model
        self.num_filters_to_prune = num_pruned
        self.scale_dict = OrderedDict()
        self.importance_dict = OrderedDict()

    def get_pruning_plan(self):
        self.compute_bn()
        self.get_importance()
        return self.get_pruning_new()

    def compute_bn(self):
        module_list = list(self.model.modules())
        for index, module in enumerate(module_list):
            if index == 1 or (isinstance(module, nn.Conv2d) and isinstance(module_list[index-1], nn.BatchNorm2d)):
                layer = index
            if isinstance(module, nn.BatchNorm2d):
                scale = module.weight.data.abs().cpu().numpy()
                self.scale_dict[layer] = scale

    def get_importance(self):
        #print(self.scale_dict.keys())
        for layer, scale in self.scale_dict.items():
            #print("layer", layer)
            #print("scale", scale.shape)

            if layer in notPrune:
                continue
            elif not pruneTable[layer]:
                self.importance_dict[layer] = scale
            else:
                for sub in pruneTable[layer]:
                    length = scale.shape[0]
                    for i in range(length):
                        if self.scale_dict[sub][i] > scale[i]:
                            scale[i] = self.scale_dict[sub][i]
                self.importance_dict[layer] = scale

    def get_pruning(self, num_filters_to_prune):
        #for layer, importance in self.importance_dict.items():
         #   print("layer", layer)
          #  print("importance", importance)

        # print("correlation:", self.correlation_dict.keys())
        # print("importance:",self.importance_dict.keys())
        filters_to_prune_per_layer = {}
        i = 0
        while i < num_filters_to_prune:
            argmin_within_layers = list(map(np.argmin, list(self.importance_dict.values())))
            min_within_layers = list(map(np.min, list(self.importance_dict.values())))
            argmin_cross_layers = np.argmin(np.array(min_within_layers))
            cut_layer_name = list(self.importance_dict.keys())[int(argmin_cross_layers)]
            cut_layer_index = list(self.importance_dict.keys())[int(argmin_cross_layers)]
            #cut_layer_index = list(self.scale_dict.keys())[int(argmin_cross_layers)]
            cut_channel_index = argmin_within_layers[int(argmin_cross_layers)]

            self.importance_dict[cut_layer_name][cut_channel_index] = 100
            if cut_layer_index not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[cut_layer_index] = []
            filters_to_prune_per_layer[cut_layer_index].append(cut_channel_index)
            i += 1

            if pruneTable[cut_layer_index]:
                for sub in pruneTable[cut_layer_index]:
                    if sub not in filters_to_prune_per_layer:
                        filters_to_prune_per_layer[sub] = []
                    filters_to_prune_per_layer[sub].append(cut_channel_index)
                    i += 1

        #for layer,channel in filters_to_prune_per_layer.items():
         #   print("cut_layer_index", layer)
          #  print("cut_channel_index", channel)

        pruned = 0
        cfg = []
        cfg_mask = []
        for layer, module in enumerate(self.model.modules()):
            if isinstance(module, nn.BatchNorm2d):
                #print("layer", layer-1)
                mask = torch.ones(module.weight.shape[0])
                if layer-1 in filters_to_prune_per_layer.keys():
                    for i in filters_to_prune_per_layer[layer-1]:
                        mask[i] = 0
                mask = mask.cuda()
                ############whole layer is cut
                if int(torch.sum(mask)) == 0:
                    mask[i] = 1.
                #print("mask", mask)

                module.weight.data.mul_(mask)
                module.bias.data.mul_(mask)
                pruned = pruned + mask.shape[0] - torch.sum(mask)
                cfg.append(int(torch.sum(mask)))
                cfg_mask.append(mask.clone())
                print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                    format(layer-1, mask.shape[0], int(torch.sum(mask))))
            elif isinstance(module, nn.MaxPool2d):
                cfg.append('M')

        return pruned, cfg, cfg_mask

    def get_pruning_new(self):
        #for layer, importance in self.importance_dict.items():
         #   print("layer", layer)
          #  print("importance", importance)

        filters_to_prune_per_layer = {}
        for layer, importance in self.importance_dict.items():
            #print("layer", layer)
            #print("importance", importance.shape[0])
            for i in range(importance.shape[0]):
                if importance[i] < 1e-3:
                    if layer not in filters_to_prune_per_layer:
                        filters_to_prune_per_layer[layer] = []
                    filters_to_prune_per_layer[layer].append(i)

                    if pruneTable[layer]:
                        for sub in pruneTable[layer]:
                            if sub not in filters_to_prune_per_layer:
                                filters_to_prune_per_layer[sub] = []
                            filters_to_prune_per_layer[sub].append(i)

        #for layer,channel in filters_to_prune_per_layer.items():
         #   print("cut_layer_index", layer)
          #  print("cut_channel_index", channel)

        pruned = 0
        cfg = []
        cfg_mask = []
        module_list = list(self.model.modules())
        for index, module in enumerate(self.model.modules()):
            if index == 1 or (isinstance(module, nn.Conv2d) and isinstance(module_list[index-1], nn.BatchNorm2d)):
                layer = index
            if isinstance(module, nn.BatchNorm2d):
                mask = torch.ones(module.weight.shape[0])
                if layer in filters_to_prune_per_layer.keys():
                    for i in filters_to_prune_per_layer[layer]:
                        mask[i] = 0
                mask = mask.cuda()
                ############whole layer is cut
                if int(torch.sum(mask)) == 0:
                    mask[i] = 1.
                #print("mask", mask)

                module.weight.data.mul_(mask)
                module.bias.data.mul_(mask)
                pruned = pruned + mask.shape[0] - torch.sum(mask)
                cfg.append(int(torch.sum(mask)))
                cfg_mask.append(mask.clone())
                print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                    format(layer, mask.shape[0], int(torch.sum(mask))))
            elif isinstance(module, nn.MaxPool2d):
                cfg.append('M')

        return pruned, cfg, cfg_mask

num_pruned = int(total * args.percent)
pruner = Pruner(model, num_pruned)
pruned, cfg, cfg_mask = pruner.get_pruning_plan()
print(cfg)
pruned_ratio = pruned/total
print('Pre-processing Successful!')

# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test(model):
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    if config.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(config.data_dir, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=config.test_batch_size, shuffle=False, **kwargs)
    elif config.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(config.data_dir, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=config.test_batch_size, shuffle=False, **kwargs)
    else:
        raise ValueError("No valid dataset is given.")
    model.eval()
    correct = 0
    for data, target in test_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct.item() / float(len(test_loader.dataset))

acc = test(model)

new_model = models.ResNet_v2(depth=config.depth, cfg=cfg, num_classes=config.num_classes)
new_model.lambda_block = checkpoint['lambda']
if use_cuda:
    new_model.cuda()

num_parameters = sum([param.nelement() for param in new_model.parameters()])
savepath = os.path.join(args.save, "prune.txt")
with open(savepath, "w") as fp:
    fp.write("Configuration: \n"+str(cfg)+"\n")
    fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
    fp.write("Test accuracy: \n"+str(acc))

old_modules = list(model.modules())
new_modules = list(new_model.modules())
layer_id_in_cfg = 0
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_id_in_cfg]
down_count = 0
n = (config.depth-2)//9
for layer_id in range(len(old_modules)):
    m0 = old_modules[layer_id]
    m1 = new_modules[layer_id]
    #print("layer ", layer_id)
    #print("old", m0)
    #print("new", m1)
    if isinstance(m0, nn.BatchNorm2d):
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        if idx1.size == 1:
            idx1 = np.resize(idx1,(1,))
        m1.weight.data = m0.weight.data[idx1.tolist()].clone()
        m1.bias.data = m0.bias.data[idx1.tolist()].clone()
        m1.running_mean = m0.running_mean[idx1.tolist()].clone()
        m1.running_var = m0.running_var[idx1.tolist()].clone()
        layer_id_in_cfg += 1
        start_mask = end_mask.clone()
        if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
            end_mask = cfg_mask[layer_id_in_cfg]
    elif isinstance(m0, nn.Conv2d):
        if layer_id == 1 or isinstance(old_modules[layer_id - 1], nn.BatchNorm2d):
            # This covers the convolutions in the residual block.
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()

            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
            continue
        # We need to consider the case where there are downsampling convolutions.
        else:
            down_start = cfg_mask[0 + down_count*n*3]
            down_end = cfg_mask[n*3+ down_count*n*3]
            idx0 = np.squeeze(np.argwhere(np.asarray(down_start.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(down_end.cpu().numpy())))
            # print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()

            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
            #print(m1.weight.data.shape)
            down_count += 1
    elif isinstance(m0, nn.Linear):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))

        m1.weight.data = m0.weight.data[:, idx0].clone()
        m1.bias.data = m0.bias.data.clone()

torch.save({'cfg': cfg, 'lambda': model.lambda_block,
            'state_dict': new_model.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))

print(new_model)
model = new_model
test(model)