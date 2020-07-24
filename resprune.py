import torch
import torch.nn as nn
import torch.nn.functional as F
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
    parser.add_argument('--save_path', default='./exp/bn/prune', type=str, metavar='PATH',
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
#print(model)

# prune
total = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        total += m.weight.data.shape[0]

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
            prune_table[cap].append(layer + 9)
            not_prune.append(layer + 9)
            for i in range(n-1):
                prune_table[cap].append(layer + 10 + i*8 + 6)
                not_prune.append(layer + 10 + i*8 + 6)
    return prune_table, not_prune
pruneTable,notPrune = get_pt()
print(pruneTable)
print(notPrune)

class Pruner:
    def __init__(self, model, num_to_prune):
        self.model = model
        self.num_filters_to_prune = num_to_prune
        self.bn_dict = OrderedDict()
        self.l1_dict = OrderedDict()
        self.correlation_dict = OrderedDict()
        self.importance_dict = OrderedDict()

    def get_pruning_plan(self):
        #self.compute_bn()
        #self.compute_l1()
        self.compute_correlation()
        self.get_importance()
        return self.get_pruning(self.num_filters_to_prune)

    def compute_bn(self):
        module_list = list(self.model.modules())
        for index, module in enumerate(module_list):
            if index == 1 or (isinstance(module, nn.Conv2d) and isinstance(module_list[index-1], nn.BatchNorm2d)):
                layer = index
            if isinstance(module, nn.BatchNorm2d):
                scale = module.weight.data.abs().cpu().numpy()
                self.bn_dict[layer] = scale

    def compute_l1(self):
        module_list = list(self.model.modules())
        for index, module in enumerate(module_list):
            if isinstance(module, nn.Conv2d):
                weights = module.weight.data.abs().clone().cpu().numpy()
                l1 = np.sum(weights, axis=(1, 2, 3))
                self.bn_dict[index] = l1

    def compute_correlation(self):
        module_list = list(self.model.modules())
        for index, module in enumerate(self.model.modules()):
            if index == 1 or (isinstance(module, nn.Conv2d) and isinstance(module_list[index-1], nn.BatchNorm2d)):
                layer = index
                for i in range(index + 1, len(module_list)):
                    if (isinstance(module_list[i], torch.nn.modules.conv.Conv2d) and isinstance(module_list[i-1], nn.BatchNorm2d)) \
                            or isinstance(module_list[i], torch.nn.modules.Linear):
                        sub_module = module_list[i]
                        break
                #print("layer: ", layer)
                #print("module: ", module)
                #print("sub_module: ", sub_module)
                weights = sub_module.weight.data.cpu().numpy()
                #print(weights.shape)
                # dense could be seen as conv whose kernel is [1,1]
                if len(weights.shape) == 2:
                    weights = np.reshape(weights, list(weights.shape) + [1, 1])

                shape = weights.shape
                # print("weights shape: ", shape)  # of shape output_channel, input_channel, h, w
                weights = np.reshape(weights, [shape[0], shape[1], shape[2] * shape[3]])  # combine h and w
                weights = np.transpose(weights, [2, 1, 0])  # of shape h*w, input_channel, output_channel
                new_shape = weights.shape
                feature_mean = np.mean(weights, axis=2, keepdims=True)  # h*w, input_channel, 1
                feature_std = np.std(weights, axis=2, keepdims=True)  # h*w, input_channel, 1
                feature = weights - feature_mean  # of output_channel, input_channel, shape h*w
                feature_t = np.transpose(feature, [0, 2, 1])
                feature_std_t = np.transpose(feature_std, [0, 2, 1])
                # corr: of shape input_channel, input_channel, h*w
                corr = np.matmul(feature, feature_t) / new_shape[2] / (np.matmul(feature_std, feature_std_t) + 1e-8)
                corr = np.abs(corr)
                mean_corr = np.mean(corr, axis=0, keepdims=False)  # of shape input_channel, input_channel
                #print(mean_corr.shape)

                norm_func = lambda x: np.unique(x)[-1]  # max norm
                # norm_func = lambda stat: np.sum(np.abs(stat)) / 2 # l1norm
                # norm_func = lambda stat: numpy.linalg.norm(stat) #l2norm
                tmp_corr = mean_corr * (1 - np.eye(mean_corr.shape[0]))  # set the numbers in principal diagonal to 0
                normalizer = norm_func(tmp_corr)
                norm_corr = mean_corr / normalizer
                #print(norm_corr.shape)

                tri_corr = np.tril(norm_corr, 0)
                sort_corr = np.sort(tri_corr, axis=0)
                # print("sort_corr", sort_corr)
                sort_corr = sort_corr[-2]
                # print("sort_corr", sort_corr)
                self.correlation_dict[layer] = sort_corr
                # importance = np.mean(sort_corr, axis=0, keepdims=False)
                importance_cor = 1 - sort_corr
                #print(importance_cor.shape)
                self.bn_dict[layer] = importance_cor

    def get_importance(self):
        #print(self.bn_dict.keys())
        for layer, scale in self.bn_dict.items():
            #print("layer", layer)
            #print("scale", scale.shape)

            if layer in notPrune:
                continue
            elif not pruneTable[layer]:
                self.importance_dict[layer] = scale
            else:
                for sub in pruneTable[layer]:
                    if sub in self.bn_dict.keys():
                        length = scale.shape[0]
                        for i in range(length):
                            if self.bn_dict[sub][i] > scale[i]:
                                scale[i] = self.bn_dict[sub][i]
                self.importance_dict[layer] = scale

    def get_pruning(self, num_filters_to_prune):
        #for layer, importance in self.importance_dict.items():
         #   print("layer", layer)
         #   print("importance", importance.shape)

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
        #    print("cut_layer_index", layer)
        #    print("cut_channel_index", channel)

        pruned = 0
        cfg = []
        cfg_mask = []
        module_list = list(self.model.modules())
        for index, module in enumerate(self.model.modules()):
            if index == 1 or (isinstance(module, nn.Conv2d) and isinstance(module_list[index-1], nn.BatchNorm2d)):
                layer = index
            if isinstance(module, nn.BatchNorm2d):
                #print("layer", layer-1)
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
                    format(layer-1, mask.shape[0], int(torch.sum(mask))))
            elif isinstance(module, nn.MaxPool2d):
                cfg.append('M')

        return pruned, cfg, cfg_mask

num_to_prune = int(total * args.percent)
pruner = Pruner(model, num_to_prune)
pruned, cfg, cfg_mask = pruner.get_pruning_plan()
print(cfg)
pruned_ratio = pruned/total
print('Pre-processing Successful!')

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).data.item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct.item() / len(test_loader.dataset)
acc = test()

# new model
new_model = models.ResNet_v2(depth=config.depth, cfg=cfg, num_classes=config.num_classes)
if use_cuda:
    new_model.cuda()

num_parameters = sum([param.nelement() for param in new_model.parameters()])
savepath = os.path.join(args.save_path, "prune.txt")
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

torch.save({'cfg': cfg, 'state_dict': new_model.state_dict()},
           os.path.join(args.save_path, 'pruned.pth.tar'))

print(new_model)
model = new_model
test()