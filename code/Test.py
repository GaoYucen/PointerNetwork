"""

Pytorch implementation of Pointer Network.

https://github.com/GaoYucen/Ptr-net

"""
#%%
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
import pickle

# 添加路径
import sys
sys.path.append('Ptr-net/code/')
from PointerNet import PointerNet
from Data_Generator import TSPDataset
from config import get_config


# if __name__ == '__main__':

params, _ = get_config()

#%% 构造model实例
if params.gpu and torch.cuda.is_available():
    USE_CUDA = True
    print('Using GPU, %i devices.' % torch.cuda.device_count())
else:
    USE_CUDA = False

model = PointerNet(params.embedding_size,
                   params.hiddens,
                   params.nof_lstms,
                   params.dropout,
                   params.bidir)

# #%%
# # 原参数 train_size = 1000000, test_size = 1000, nof_epoch = 50000
# train_dataset = TSPDataset(params.train_size,
#                      params.nof_points)
#
# # 存储dict dataset
# with open('Ptr-net/data/train.pkl', 'wb') as f:
#     pickle.dump(train_dataset, f)
#
# test_dataset = TSPDataset(params.test_size,
#                             params.nof_points)
#
# with open('Ptr-net/data/test.pkl', 'wb') as f:
#     pickle.dump(test_dataset, f)

#%% 读取dict dataset
# with open('Ptr-net/data/train.pkl', 'rb') as f:
#     dataset = pickle.load(f)

dataset = TSPDataset(params.train_size,
                        params.nof_points)

dataloader = DataLoader(dataset,
                        batch_size=params.batch_size,
                        shuffle=True,
                        num_workers=0)

#%% Define loss
if USE_CUDA:
    model.cuda()
    net = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

CCE = torch.nn.CrossEntropyLoss()
model_optim = optim.Adam(filter(lambda p: p.requires_grad,
                                model.parameters()),
                         lr=params.lr)

losses = []

#%%
for epoch in range(params.nof_epoch):
    batch_loss = []
    iterator = tqdm(dataloader, unit='Batch')

    for i_batch, sample_batched in enumerate(iterator):
        iterator.set_description('Epoch %i/%i' % (epoch+1, params.nof_epoch))

        train_batch = Variable(sample_batched['Points'].float())
        target_batch = Variable(sample_batched['Solution'])

        if USE_CUDA:
            train_batch = train_batch.cuda()
            target_batch = target_batch.cuda()

        o, p = model(train_batch)
        o = o.contiguous().view(-1, o.size()[-1])

        target_batch = target_batch.view(-1)

        loss = CCE(o, target_batch)
        losses.append(loss.data.item())
        batch_loss.append(loss.data.item())

        model_optim.zero_grad()
        loss.backward()
        model_optim.step()

        iterator.set_postfix(loss='{}'.format(loss.data.item()))

    iterator.set_postfix(loss=sum(batch_loss)/len(batch_loss))

#%%
torch.save(model.state_dict(), 'param/param.pkl')
print('save success')

#%% 测试模型
model.load_state_dict(torch.load('param/param.pkl'))
print('load success')

# 读取测试数据
with open('Ptr-net/data/test.pkl', 'rb') as f:
    test_dataset = pickle.load(f)

test_dataloader = DataLoader(test_dataset,
                            batch_size=params.batch_size,
                            shuffle=True,
                            num_workers=0)

#%% 测试
if USE_CUDA:
    model.cuda()
    net = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

CCE = torch.nn.CrossEntropyLoss()

losses = []
batch_loss = []

iterator = tqdm(test_dataloader, unit='Batch')

for i_batch, sample_batched in enumerate(iterator):
    test_batch = Variable(sample_batched['Points'])
    target_batch = Variable(sample_batched['Solution'])

    if USE_CUDA:
        test_batch = test_batch.cuda()
        target_batch = target_batch.cuda()

    o, p = model(test_batch)
    o = o.contiguous().view(-1, o.size()[-1])

    target_batch = target_batch.view(-1)

    loss = CCE(o, target_batch)

    losses.append(loss.data.item())
    batch_loss.append(loss.data.item())

    iterator.set_postfix(loss='{}'.format(loss.data.item()))

iterator.set_postfix(loss=np.average(batch_loss))

