import os
import json
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import lpmm
import math
import cv2 
import numpy as np
import logging
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from xml.dom.minidom import parse
import xml.dom.minidom
from tqdm import tqdm
from sklearn.metrics import f1_score
import argparse
from torch.nn.modules.loss import _Loss
from torch.utils.tensorboard import SummaryWriter
from efficientnet_pytorch import EfficientNet
from resnet import ResNet34, ResNet18
from torchvision.models import resnet34
from torchtoolbox.tools import mixup_data, mixup_criterion
from util import *
from moganet import *


# 设置随机数种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class custom_dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, mode, transform=None):
        super().__init__()
        
        self.data_path = data_path
        self.transform = transform
        self.mode = mode
        
        self.class_dict = {}
        idx = 0
        
        self.train_data = []
        self.train_label = []

        self.test_data = []
        self.test_label = []
        
        test_num = [30, 5, 1]
        total_num = 0
        
        for i, v in enumerate(os.walk(data_path).__next__()[1]):
            v_path = os.path.join(data_path, v)   # /home/data/xxxx
            
            class_name = os.walk(os.path.join(data_path, v)).__next__()[1]
            
            for name in class_name:
                if name not in self.class_dict:
                    DOMTree = xml.dom.minidom.parse(os.path.join(v_path, f'{name}.xml'))
                    collection = DOMTree.documentElement
                    root = collection.getElementsByTagName('class')[0]
                    self.class_dict[name] = (idx, root.childNodes[0].data)
                    idx += 1
            
            for name in class_name:
                images_path = os.path.join(v_path, name)
                files = [os.path.join(images_path, file) for file in os.walk(images_path).__next__()[2]]

                total_num += len(files)
                # print(f'Total number of images in {v}/{name}: {len(files)}')

                test_data = random.sample(files, k=test_num[i])
                train_data = list(set(files) - set(test_data))

                self.train_data += train_data
                self.test_data += test_data

                self.train_label += [self.class_dict[name][0]] * len(train_data)
                self.test_label += [self.class_dict[name][0]] * len(test_data)

        print('Total number of data:', total_num)
        if self.mode == 'train':
            self.data = self.train_data
            self.label = self.train_label
        elif self.mode == 'test':
            self.data = self.test_data
            self.label = self.test_label
        else:
            raise ValueError('没有选择模式 [train, test]')
        
    def __getitem__(self, index):
        if self.transform is None:
            return Image.open(self.data[index]).convert('RGB'), self.label[index]
        else:
            image = cv2.imread(self.data[index])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transformed = self.transform(image=image)
            image = transformed["image"]
            # return self.transform(image=Image.open(self.data[index]).convert('RGB')), self.label[index]
            return image, self.label[index]
    
    def __len__(self):
        return len(self.data)


class custom_model(nn.Module):
    def __init__(self, backbone, hidden_dim, nb_class):
        super(custom_model, self).__init__()

        # self.backbone = torch.nn.Sequential(*(list(backbone.children())[:-1]))
        # self.backbone = backbone
        # self.fc = torch.nn.Linear(hidden_dim, nb_class)
        backbone.head = nn.Identity()
        self.backbone = backbone
        self.fc = ArcMarginProduct(hidden_dim, nb_class, 30, 0.3, False, 0.0)

    def forward(self, x, y):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # [128, 3, 32, 32]
        output = self.fc(x, y)
        return output

    
# From https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/blob/master/src/modeling/metric_learning.py
# Added type annotations, device, and 16bit support
class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta + m)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float,
        m: float,
        easy_margin: bool,
        ls_eps: float,
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input: torch.Tensor, label: torch.Tensor, device: str = "cuda") -> torch.Tensor:
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # Enable 16 bit precision
        cosine = cosine.to(torch.float32)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

    
class FocalLoss(nn.Module):
    def __init__(self, gamma=1, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


# 替换focal loss，可以测试polyloss
class PolyLoss(_Loss):
    def __init__(self, softmax, ce_weight=None, reduction='mean', epsilon=1.0):
        super().__init__()
        self.softmax = softmax
        self.reduction = reduction
        self.epsilon = epsilon
        self.cross_entropy = nn.CrossEntropyLoss(weight=ce_weight, reduction='none')

    def forward(self, input, target):
        if len(input.shape) - len(target.shape) == 1:
            target = target.unsqueeze(1).long()
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch != n_target_ch:
            self.ce_loss = self.cross_entropy(input, torch.squeeze(target, dim=1).long())
            target = self.to_one_hot(target, num_classes=n_pred_ch)
        else:
            self.ce_loss = self.cross_entropy(input, torch.argmax(target, dim=1))

        if self.softmax:
            input = torch.softmax(input, 1)

        pt = (input * target).sum(dim=1) 
        
        poly_loss = self.ce_loss + self.epsilon * (1 - pt)

        polyl = torch.mean(poly_loss)  # the batch and channel average
        # polyl = torch.sum(poly_loss)  # sum over the batch and channel dims
        return (polyl)
    
    def to_one_hot(self, labels, num_classes, dtype=torch.float, dim=1):
        if labels.ndim < dim + 1:
            shape = list(labels.shape) + [1] * (dim + 1 - len(labels.shape))
            labels = torch.reshape(labels, shape)
        sh = list(labels.shape)
        sh[dim] = num_classes
        o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
        labels = o.scatter_(dim=dim, index=labels.long(), value=1)
        return labels


def main(pretrain, seed, train_batch, test_batch, lr, epochs, img_size):
        # log日志文件
    writer = SummaryWriter(f'/project/train/tensorboard/')       # 可视化训练过程
    
    # 构建dataloader
    data_path = '/home/data/'
    
    # train_transform = transforms.Compose([
    #     transforms.Resize((img_size, img_size)),
    #     transforms.RandomRotation(15),
    #     transforms.RandomCrop(img_size, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    # ])
    
    train_transform = A.Compose([
         A.CLAHE(clip_limit=40, tile_grid_size=(10, 10),p=1.0),
         A.Resize(img_size, img_size),
         A.HorizontalFlip(p=0.5),
         A.VerticalFlip(p=0.5),
         A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
         A.OneOf([A.RandomBrightness(limit=0.1, p=1), A.RandomContrast(limit=0.1, p=1)]),
         A.Normalize(),
         ToTensorV2(p=1.0),
     ])
    
    # test_transform = transforms.Compose([
    #     transforms.Resize((img_size, img_size)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    # ])
    
    test_transform = A.Compose([
         A.CLAHE(clip_limit=40, tile_grid_size=(10, 10),p=1.0),
         A.Resize(img_size, img_size),
         A.Normalize(),
         ToTensorV2(p=1.0),
     ])
    
    train_datasets = custom_dataset(data_path, mode='train', transform=train_transform)
    print('Total number of train data:', len(train_datasets))
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=train_batch, shuffle=True)
    
    val_datasets = custom_dataset(data_path, mode='test', transform=test_transform)
    print('Total number of test data:', len(val_datasets))
    val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=test_batch, shuffle=False)
    
    # save class
    classes_dict = {v[0]:v[1] for k, v in train_datasets.class_dict.items()}
    with open('/project/train/models/classes.json', 'w') as f:
        json.dump(classes_dict, f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setup_seed(seed)

    # model = custom_model(backbone, 2048, len(train_datasets.class_dict))
    # model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=len(train_datasets.class_dict))
    # backbone = ResNet18(input_channels=3, nb_class=len(train_datasets.class_dict))
    # backbone = moganet_tiny(pretrained=False)
    backbone = ghostnetv2(num_classes=len(train_datasets.class_dict), width=1.0, dropout=0.0, args=None)
    model = custom_model(backbone, 256, len(train_datasets.class_dict))
    if os.path.exists('/project/train/models/pests_2023.pth'):
        print('Pretrained model exists.')
        model.load_state_dict(torch.load('/project/train/models/pests_2023.pth', map_location='cpu'))
    model.to(device)
    
    print(model)
    loss_fn = FocalLoss()
    loss_fn.to(device)

    # weight_decay=1e-3

    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=1e-4)
    optimizer = lpmm.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=5e-4)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[175, 250, 375], gamma=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-5, last_epoch=-1)

    # print(model)
    accuracy = 0

    for epoch in range(epochs):
        model, f1score, train_loss = train(model, optimizer, train_dataloader, epoch, device, loss_fn, scheduler)
        correct, test_loss = test(model, val_dataloader, device)
        
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', f1score, epoch)
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', correct, epoch)
        
        if correct > accuracy:
            save_name = f"pests_{seed}.pth"
            accuracy = correct
            torch.save(model.state_dict(), f'/project/train/models/{save_name}')
    writer.close()


def train(model, optimizer, train_loader, epoch, device, loss_fn, scheduler=None):
    model.train()
    train_loss = 0
    correct = 0
    
    prob_all = []
    label_all = []

    for data, target in tqdm(train_loader, desc=f"epoch_{epoch}"):
        rand = np.random.rand()
        
        data = data.to(device)
        target = target.to(device)

        data, target = Variable(data), Variable(target)  # 把数据转换成Variable
        
        # use mix up and cut up 
        if rand < 0.25:
            data, labels_a, labels_b, lam = mixup_data(data, target, 1)
            output = model(data, target)
            loss = mixup_criterion(loss_fn, output, labels_a, labels_b,lam)
        elif rand < 0.5 and rand > 0.25:
            lam = np.random.beta(1, 1)
            rand_index = torch.randperm(data.size()[0])
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
            data[:, :, bbx1:bbx2, bby1:bby2] = data[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
            # compute output
            output = model(data, target)
            loss = loss_fn(output, target_a) * lam + loss_fn(output, target_b) * (1. - lam)
        else:
            output = model(data, target)
            loss = loss_fn(output, target)
        
        optimizer.zero_grad()  # 优化器梯度初始化为零
        # output = model(data)  # 把数据输入网络并得到输出，即进行前向传播
        # loss = F.cross_entropy(output, target)  # 交叉熵损失函数

        train_loss += loss.item() * data.shape[0]  # 计算训练误差
        loss.backward(retain_graph=True)  # 反向传播梯度
        optimizer.step()  # 结束一次前传+反传之后，更新参数
        pred = output.data.max(1, keepdim=True)[1]  # 获取预测值
        prob_all.extend(pred.cpu().detach().numpy().reshape(-1))
        label_all.extend(target.cpu().detach().numpy().reshape(-1))
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # 计算预测正确数

    scheduler.step()

    f1_scores = f1_score(label_all, prob_all, average='micro')
    
    logging.info("F1-Score:{:.4f}".format(f1_scores))
    print("Train F1-Score:{:.4f}".format(f1_scores))
    logging.info("\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
        train_loss / len(train_loader.dataset), correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)
    ))
    print("\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
        train_loss / len(train_loader.dataset), correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)
    ))
    return model, f1_scores, train_loss / len(train_loader.dataset)


def test(model, loader, device):
    model.eval()                                            # 设置为test模式
    test_loss = 0                                           # 初始化测试损失值为0
    correct = 0                                             # 初始化预测正确的数据个数为0
    
    prob_all = []
    label_all = []
    
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        data, target = Variable(data), Variable(target)     # 计算前要把变量变成Variable形式，因为这样子才有梯度

        with torch.no_grad():
            output = model(data, target)

        loss = F.cross_entropy(output, target.long())  # sum up batch loss 把所有loss值进行累加
        test_loss += loss.item() * data.shape[0]

        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        
        prob_all.extend(pred.cpu().detach().numpy().reshape(-1))
        label_all.extend(target.cpu().detach().numpy().reshape(-1))
        
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # 对预测正确的数据个数进行累加
        
    test_loss /= len(loader.dataset)  # 因为把所有loss值进行过累加，所以最后要除以总得数据长度才得平均loss

    f1_scores = f1_score(label_all, prob_all, average='micro')
    
    logging.info("F1-Score:{:.4f}".format(f1_scores))
    print("Test F1-Score:{:.4f}".format(f1_scores))
    logging.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(loader.dataset),
        100. * correct / len(loader.dataset)))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(loader.dataset),
        100. * correct / len(loader.dataset)))
    return correct, test_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, default='efficientnet_b2')
    parser.add_argument('--pretrain', type=bool, default=True)
    parser.add_argument('--train_batch', type=int, default=150)
    parser.add_argument('--test_batch', type=int, default=150)
    parser.add_argument('--img_size', type=int, default=112)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.09)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--epochs', type=int, default=50)

    args = parser.parse_args()
    main(args.pretrain, args.seed, args.train_batch, args.test_batch, args.lr, args.epochs, args.img_size)

