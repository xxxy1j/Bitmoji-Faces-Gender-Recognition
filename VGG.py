import os
import pandas as pd
import torch
import torchvision
from torch import nn
import datetime

def read_csv_labels(fname):
    """读取fname来给标签字典返回一个文件名"""
    with open(fname, 'r') as f:
        # 跳过文件头行(列名)
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

def vgg(conv_arch):
    conv_blks = []
    in_channels = 3
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 2))

def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            net.eval()  # 评估模式, 这会关闭dropout
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            net.train()  # 改回训练模式
            n += y.shape[0]
    return acc_sum / n

def train_batch(net, X, y, loss, trainer, device):
    X = X.to(device)
    y = y.to(device)
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = (torch.argmax(pred, dim=1) == y).sum().item() / y.size(0)
    return train_loss_sum, train_acc_sum

def train(net, train_iter, valid_iter, num_epochs, lr, wd, device):
    #trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    trainer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=wd)
    starttime = datetime.datetime.now()
    for epoch in range(num_epochs):
        train_loss, train_acc = 0.0, 0.0
        sum_count = 0
        net.cuda()
        net.train()
        for i, (features, labels) in enumerate(train_iter):
            l, acc = train_batch(net, features, labels, loss, trainer, device)
            train_loss += l
            train_acc += acc
            sum_count += labels.shape[0]
        if valid_iter is not None:
            valid_acc = evaluate_accuracy(valid_iter, net, None)
        print(f'Epoch{epoch}: train loss {train_loss / sum_count:.3f}, 'f'train acc {train_acc/(i+1):.3f}')
        if valid_iter is not None:
            print(f'valid acc {valid_acc:.3f}')
    endtime = datetime.datetime.now()
    print(f'Runing_time:{(endtime - starttime).seconds}')

if __name__ == '__main__':
    #data_dir = '/mnt/pycharm_project_4/BitmojiDataset_Sample/'
    data_dir = '/home/BitmojiDataset_Sample'
    labels = read_csv_labels(os.path.join(data_dir, 'train.csv'))
    batch_size = 32
    transform_train = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)), torchvision.transforms.RandomHorizontalFlip(),torchvision.transforms.ToTensor()])
    transform_test = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)), torchvision.transforms.ToTensor()])
    train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train_valid_test', folder), transform=transform_train)for folder in ['train', 'train_valid']]
    valid_ds, test_ds = [torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train_valid_test', folder), transform=transform_test)for folder in ['valid', 'test']]
    train_iter, train_valid_iter = [torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, drop_last=True)for dataset in (train_ds, train_valid_ds)]
    valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,drop_last=True)
    test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,drop_last=False)
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    net = vgg(conv_arch)
    #print(net.cuda())
    loss = nn.CrossEntropyLoss(reduction="none")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs, lr, wd = 20, 2e-4, 5e-4
    train(net, train_iter, valid_iter, num_epochs, lr, wd, device)
    preds = []
    train(net, train_valid_iter, None, num_epochs, lr, wd, device)
    for X, _ in test_iter:
        y_hat = net(X.to(device))
        preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())
    sorted_ids = list(range(3000, len(test_ds) + 3000))
    df = pd.DataFrame({'image_id': sorted_ids, 'is_male': preds})
    df['image_id'] = df['image_id'].apply(lambda x: str(x) + '.jpg')
    df['is_male'] = df['is_male'].apply(lambda x: train_valid_ds.classes[x])
    df.to_csv('submission_vgg.csv', index=False)