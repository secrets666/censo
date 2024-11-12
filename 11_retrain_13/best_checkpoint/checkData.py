import numpy as np
import torch
import matplotlib.pyplot as plt

weight = torch.from_numpy(np.array([1.5]*4 + [2]*7 + [3]*7 + [4]*6) * np.log(np.arange(24)+1))

def score(y_pred, y_true, acc_weight):
    # for pytorch
    # acc_weight = np.array([1.5]*4 + [2]*7 + [3]*7 + [4]*6) * np.log(np.arange(24)+1)
    # acc_weight = torch.from_numpy(acc_weight).to(device)
    pred = y_pred - y_pred.mean(dim=0, keepdim=True)  # (N, 24)
    true = y_true - y_true.mean(dim=0, keepdim=True)  # (N, 24)
    cor = (pred * true).sum(dim=0) / (torch.sqrt(torch.sum(pred**2, dim=0) * torch.sum(true**2, dim=0)) + 1e-6)
    acc = (acc_weight * cor).sum()
    rmse = torch.mean((y_pred - y_true)**2, dim=0).sqrt().sum()
    print("rmse=", torch.mean((y_pred - y_true)**2, dim=0).sqrt())
    return 2/3. * acc - rmse, cor


pred = np.load(r"GODAS_pred.npy")
tru = np.load(r"GODAS_true.npy")

print(pred.shape)  # (395,38,2,24,48)
print(tru.shape)  # (395,38,2,24,48)

pred = torch.from_numpy(pred)
tru = torch.from_numpy(tru)

outputs = pred[:,1:,:,:,:] # (N, 37, H, W)
nino_pred = outputs[:, -26:, 0, 10:13, 19:30].mean(dim=[2, 3])  # (N, 26)
nino_pred = nino_pred.unfold(dimension=1, size=3, step=1).mean(dim=2)  # (N, 24)

outputs = tru[:,1:,:,:,:] # (N, 37, H, W)
nino_true = outputs[:, -26:, 0, 10:13, 19:30].mean(dim=[2, 3])  # (N, 26)
nino_true = nino_true.unfold(dimension=1, size=3, step=1).mean(dim=2)  # (N, 24)

print(nino_pred.shape)  # (395,24)
print(nino_true.shape)  # (395,24)

sc, cor = score(nino_pred, nino_true, weight)
print(sc)
print(cor)

print("***********"*5)

nino4_area_pred = pred[:,-26:,0,10:13,23:36]  # nino3 region
nino4_area_true = tru[:,-26:,0,10:13,23:36]

nino4_pred = nino4_area_pred.mean(dim=[-1, -2])  # (N, 26)
nino4_pred = nino4_pred.unfold(dimension=1, size=3, step=1).mean(dim=2)  # (N, 24)
nino4_true = nino4_area_true.mean(dim=[-1, -2])  # (N, 26)
nino4_true = nino4_true.unfold(dimension=1, size=3, step=1).mean(dim=2)  # (N, 24)


# np.save('./data/nino3_pred.npy', nino4_pred)
# np.save('./data/nino3_true.npy', nino4_true)

sc, cor = score(nino4_pred, nino4_true, weight)
print(sc)
print('******cor*****'*10)
print(cor)

nino4_area_pred = pred[:,-26:,0,10:13,13:24]  # nino4 region
nino4_area_true = tru[:,-26:,0,10:13,13:24]

nino4_pred = nino4_area_pred.mean(dim=[2, 3])  # (N, 26)
nino4_pred = nino4_pred.unfold(dimension=1, size=3, step=1).mean(dim=2)  # (N, 24)
nino4_true = nino4_area_true.mean(dim=[2, 3])  # (N, 26)
nino4_true = nino4_true.unfold(dimension=1, size=3, step=1).mean(dim=2)  # (N, 24)

sc, cor = score(nino4_pred, nino4_true, weight)
print(sc)
print('******cor*****'*10)
print(cor)
print('******rmse*****'*10)


