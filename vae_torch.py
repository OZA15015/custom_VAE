import torch
from torchvision import datasets, transforms
import copy
from torch.nn import functional as F
from torch.autograd import Variable

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))])

dataset_train = datasets.MNIST(
    '~/mnist',
    train = True,
    download = True,
    transform = transform)

dataset_valid = datasets.MNIST(
    '~/mnist',
    train = False,
    download = True,
    transform = transform)

b_size = 1000

dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                        batch_size = b_size,
                                        shuffle = True,
                                        num_workers = 4)
dataloader_valid = torch.utils.data.DataLoader(dataset_valid,
                                        batch_size = b_size,
                                        shuffle = True,
                                        num_workers = 4)
#num_workers」は複数処理をするかどうかで,2以上の場合その値だけ並行処理
#以上, MNISTデータ読み込み部

import torch.nn as nn
import torch.nn.functional as F

device = 'cuda'

class VAE(nn.Module):
    def __init__(self, z_dim):
        super(VAE, self).__init__()
        self.dense_enc1 = nn.Linear(28*28, 200)
        self.dense_enc2 = nn.Linear(200, 100)
        self.dense_encmean = nn.Linear(100, z_dim)
        self.dense_encvar = nn.Linear(100, z_dim)
        self.dense_dec1 = nn.Linear(z_dim, 100)
        self.dense_dec2 = nn.Linear(100, 200)
        self.dense_dec3 = nn.Linear(200, 28*28)

    def _encoder(self, x):
        x = F.relu(self.dense_enc1(x))
        x = F.relu(self.dense_enc2(x))
        mean = self.dense_encmean(x)
        var = self.dense_encvar(x)
        #var = F.softplus(self.dense_encvar(x))
        return mean, var

    def _sample_z(self, mean, var): #普通にやると誤差逆伝搬ができないのでReparameterization Trickを活用
        epsilon = torch.randn(mean.shape).to(device)
        #return mean + torch.sqrt(var) * epsilon #平均 + episilonは正規分布に従う乱数, torc.sqrtは分散とみなす？平均のルート
        return mean + epsilon * torch.exp(0.5*var)
        # イメージとしては正規分布の中からランダムにデータを取り出している
        #入力に対して潜在空間上で類似したデータを復元できるように学習, 潜在変数を変化させると類似したデータを生成
        #Autoencoderは決定論的入力と同じものを復元しようとする


    def _decoder(self,z):
        x = F.relu(self.dense_dec1(z))
        x = F.relu(self.dense_dec2(x))
        x = F.sigmoid(self.dense_dec3(x))
        return x

    def forward(self, x):
        mean, var = self._encoder(x)
        z = self._sample_z(mean, var)
        x = self._decoder(z)
        return x, z
    
    def loss(self, x): #lossは交差エントロピーを採用している, MSEの事例もある
        #https://tips-memo.com/vae-pytorch#i-7, http://aidiary.hatenablog.com/entry/20180228/1519828344のlossを参考
        mean, var = self._encoder(x)
        #KL = -0.5 * torch.mean(torch.sum(1 + torch.log(var) - mean**2 - var)) #オリジナル, mean意味わからんけど, あんまり値が変わらないから
        #上手くいくんじゃないか
        #KL = 0.5 * torch.sum(torch.exp(var) + mean**2 - 1. - var)
        KL = -0.5 * torch.sum(1 + var - mean.pow(2) - var.exp()) 
        # sumを行っているのは各次元ごとに算出しているため
        #print("KL: " + str(KL))
        z = self._sample_z(mean, var)
        y = self._decoder(z)
        #delta = 1e-8
        #reconstruction = torch.mean(torch.sum(x * torch.log(y + delta) + (1 - x) * torch.log(1 - y + delta)))
        reconstruction = F.binary_cross_entropy(y, x.view(-1, 784), size_average=False)
        #交差エントロピー誤差を利用して, 対数尤度の最大化を行っている, 2つのみ=(1-x), (1-y)で算出可能
        #http://aidiary.hatenablog.com/entry/20180228/1519828344(参考記事)
        #print("reconstruction: " + str(reconstruction))
        #lower_bound = [-KL, reconstruction]
        #両方とも小さくしたい, クロスエントロピーは本来マイナス, KLは小さくしたいからプラスに変換
        #returnで恐らくわかりやすくするために, 目的関数から誤差関数への変換をしている
        #return -sum(lower_bound)
        return KL + reconstruction


import numpy as np
from torch import optim

def main():
    model = VAE(2).to(device)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    model.train()
    for i in range(30): #num epochs
        losses = []
        for x, t in dataloader_train: #data, label 
            x = x.to(device)
            optimizer.zero_grad() #batchごとに勾配の更新
            y = model(x)
            loss = model.loss(x) / b_size
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())
        print("Epoch: {} loss: {}".format(i, np.average(losses)))

if __name__ == "__main__":
    main()
