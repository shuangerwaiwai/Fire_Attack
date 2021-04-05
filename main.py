from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

#train：训练模式 test：测试模式
#adv_train: 对抗训练 adv_test: 最后测试
stage='TRAIN'

#original 普通训练
#adv 对抗训练
mode = 'adv'

#数据集的位置
if mode=='original':
    data_dir = './FLAME_Data'
    saved_model = 'my_model.pth'
elif mode=='adv':
    data_dir = './ADV_DATA'
    saved_model = 'my_adv_model.pth'

#使用的模型
model_name = "resnet"

#最后的分类个数
num_classes = 2

#batch大小
batch_size = 16

#迭代的次数
num_epochs = 5

#True：特征提取  Flase：微调
feature_extract = False



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, dataloaders, critertion, optimizer, num_epochs=25, is_inception=False):
    #开始时间
    since = time.time()

    val_acc_history = []

    #验证集上表现最好的模型
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)

        phases = ['train', 'val']

        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = critertion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 保存最好的模型参数
    torch.save(best_model_wts, saved_model)
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def test_Model(model, dataloaders, critertion):
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = critertion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    loss = running_loss / len(dataloaders['test'].dataset)
    acc = running_corrects.double() / len(dataloaders['test'].dataset)

    print('Loss: {:.4f} Acc: {:.4f}'.format(loss, acc))

def set_parameters_requires_grad(model, feature_extracting):
    #如果是微调，参数需要求导;如果是特征提取，则不需要求导
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

#初始化模型
def initialize_model(model_name, num_classes, feature_extract, use_pretrained):
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        if stage=='TRAIN':
            model_ft = models.resnet18(pretrained=use_pretrained)
            set_parameters_requires_grad(model_ft, feature_extract)
        else:
            model_ft = models.resnet18(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    return model_ft, input_size

#损失函数
criterion = nn.CrossEntropyLoss()


# 需要更新的参数
def params_to_update(model):
    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print("\t", name)
    return params_to_update

#训练模式
if stage == 'TRAIN':

    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
        ])
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

    model_ft = model_ft.to(device)

    #需要更新的参数
    params_to_update = params_to_update(model_ft)

    #优化器
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))


else:
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=False)
    model_ft.to(device)
    model_ft.load_state_dict(torch.load(saved_model, map_location='cpu'))
    model_ft.eval()

    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
        ])
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['test']}
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['test']}

    test_Model(model_ft, dataloaders_dict, criterion)
