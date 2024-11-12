import numpy as np
from torch.utils.data import Dataset
import xarray as xr
from pathlib import Path
import torch


def prepare_inputs_targets(len_time, input_gap, input_length, pred_shift, pred_length, samples_gap):
    # input_gap=1: time gaps between two consecutive input frames
    # input_length=12: the number of input frames
    # pred_shift=26: the lead_time of the last target to be predicted
    # pred_length=26: the number of frames to be predicted
    assert pred_shift >= pred_length
    input_span = input_gap * (input_length - 1) + 1
    pred_gap = pred_shift // pred_length
    input_ind = np.arange(0, input_span, input_gap)
    target_ind = np.arange(0, pred_shift, pred_gap) + input_span + pred_gap - 1
    ind = np.concatenate([input_ind, target_ind]).reshape(1, input_length + pred_length)
    max_n_sample = len_time - (input_span+pred_shift-1)
    ind = ind + np.arange(max_n_sample)[:, np.newaxis] @ np.ones((1, input_length+pred_length), dtype=int)
    return ind[::samples_gap]


def fold(data, size=36, stride=12):
    # inverse of unfold/sliding window operation
    # only applicable to the case where the size of the sliding windows is n*stride
    # data (N, size, *)
    # outdata (N_, *)
    # N/size is the number/width of sliding blocks
    assert size % stride == 0
    times = size // stride
    remain = (data.shape[0] - 1) % times
    if remain > 0:
        ls = list(data[::times]) + [data[-1, -(remain*stride):]]
        outdata = np.concatenate(ls, axis=0)  # (36*(151//3+1)+remain*stride, *, 15)
    else:
        outdata = np.concatenate(data[::times], axis=0)  # (36*(151/3+1), *, 15)
    assert outdata.shape[0] == size * ((data.shape[0]-1)//times+1) + remain * stride
    return outdata


def data_transform(data, num_years_per_model):
    # data (2919, 36, 24, 72)
    # num_years_per_model: 139
    length = data.shape[0]
    assert length % num_years_per_model == 0
    num_models = length // num_years_per_model
    outdata = np.stack(np.split(data, length/num_years_per_model, axis=0), axis=-1)  # (139, 36, 27, 48, 21)
    outdata = fold(outdata)  # (1692,24,48,21)  # 起到的作用实际上就是将138年12个月的数据和第139年36个月的数据拼接在一起，得到1692个月的数据

    # check output data
    assert outdata.shape[-1] == num_models
    assert not np.any(np.isnan(outdata))
    return outdata


def read_raw_data(ds_dir, file_name, configs, out_dir=None):
    # read and process raw cmip data from CMIP_train.nc and CMIP_label.nc
    train_cmip = xr.open_dataset(Path(ds_dir) / (file_name + '_train.nc')).transpose('year', 'month', 'lat', 'lon')
    label_cmip = xr.open_dataset(Path(ds_dir) / ('label_mon/' + file_name + '_label_mon.nc')).transpose('year', 'month')

    # select longitudes
    lon = train_cmip.lon.values
    lon = lon[np.logical_and(lon>=95, lon<=330)]
    train_cmip = train_cmip.sel(lon=lon)

    if file_name == "CMIP":
        n_years = 139
    elif file_name == "SODA":
        n_years = 98
    elif file_name == 'GODAS':
        n_years = 34

    cmip6sst = data_transform(train_cmip.sst.values[:], n_years)  # train_cmip.sst.values[:]=(2919,36,24,48)  (1692,24,48,21)
    cmip6hc = data_transform(train_cmip.t300.values[:], n_years)
    cmip6embed1 = data_transform(train_cmip.month_embed1.values[:], n_years)  # .. (1692,24,48,21)
    cmip6embed2 = data_transform(train_cmip.month_embed2.values[:], n_years)  # .. (1692,24,48,21)
    # print(cmip6sst.shape)  ## (1692,24,48,21)

    cmip6sst = np.stack([cmip6sst, cmip6hc, cmip6embed1, cmip6embed2], axis=4)  # (1692,24,48,21,3)


    cmip6nino = data_transform(label_cmip.nino.values[:], n_years)  # (1692,21)
    cmip6mon = data_transform(label_cmip.mon.values[:], n_years)
    cmip6season = data_transform(label_cmip.season.values[:], n_years)

    cmip6nino = np.stack([cmip6nino, cmip6mon, cmip6season], axis=-1)
    ## (1692,21)
    assert len(cmip6sst.shape) == 5
    assert len(cmip6nino.shape) == 3

    # store processed data for faster data access
    if out_dir is not None:
        ds_cmip6 = xr.Dataset({'sst': (['month', 'lat', 'lon', 'model'], cmip6sst),
                               'nino': (['month', 'model'], cmip6nino)},
                              coords={'month': np.repeat(np.arange(1, 13)[None], cmip6nino.shape[0] // 12, axis=0).flatten(),
                                      'lat': train_cmip.lat.values, 'lon': train_cmip.lon.values,
                                      'model': np.arange(15)+1})
        ds_cmip6.to_netcdf(Path(out_dir) / 'cmip6.nc')

    train_cmip.close()
    label_cmip.close()
    return cmip6sst, cmip6nino


def read_from_nc(ds_dir):
    # an alternative for reading processed data
    cmip6 = xr.open_dataset(Path(ds_dir) / 'cmip6.nc').transpose('month', 'lat', 'lon', 'model')
    cmip5 = xr.open_dataset(Path(ds_dir) / 'cmip5.nc').transpose('month', 'lat', 'lon', 'model')
    return cmip6.sst.values, cmip5.sst.values, cmip6.nino.values, cmip5.nino.values



def score(y_pred, y_true, acc_weight):
    # for pytorch
    # acc_weight = np.array([1.5]*4 + [2]*7 + [3]*7 + [4]*6) * np.log(np.arange(24)+1)
    # acc_weight = torch.from_numpy(acc_weight).to(device)
    pred = y_pred - y_pred.mean(dim=0, keepdim=True)  # (N, 24)
    true = y_true - y_true.mean(dim=0, keepdim=True)  # (N, 24)
    cor = (pred * true).sum(dim=0) / (torch.sqrt(torch.sum(pred**2, dim=0) * torch.sum(true**2, dim=0)) + 1e-6)
    acc = (acc_weight * cor).sum()
    rmse = torch.mean((y_pred - y_true)**2, dim=0).sqrt().sum()
    return 2/3. * acc - rmse, cor


def cat_over_last_dim(data):
    ## data=(331,38,24,48,21,3)
    if len(data.shape)>=5:
        return np.concatenate(np.moveaxis(data, 4, 0), axis=0)
    else:
        return np.concatenate(np.moveaxis(data, 2, 0), axis=0)


class cmip_dataset(Dataset):
    def __init__(self, sst_cmip6, nino_cmip6, samples_gap):
        super().__init__()
        # cmip6 (1692,24,48,21,3)  nino_cmip6=(1692,21, 3)
        sst1 = []
        target_nino = []
        if sst_cmip6 is not None:
            assert len(sst_cmip6.shape) == 5
            assert len(nino_cmip6.shape) == 3
            idx_sst = prepare_inputs_targets(sst_cmip6.shape[0], input_gap=1, input_length=12,
                                             pred_shift=26, pred_length=26, samples_gap=samples_gap)
            # print(idx_sst.shape)  # (331,38)
            # print(sst_cmip6.shape)  # (1962,24,48,21,4)
            # print(nino_cmip6.shape)  # (1692,21,3)
            print(sst_cmip6[idx_sst].shape)

            sst1.append(cat_over_last_dim(sst_cmip6[idx_sst]))  # sst_cmip6[idx_sst]=(331,38,24,48,21,4)
            target_nino.append(cat_over_last_dim(nino_cmip6[idx_sst]))  # nino_cmip6[idx_sst]=(331,38,21,3)

        # sst data containing both the input and target
        sst1 = np.concatenate(sst1, axis=0)  # (N, 38, lat, lon, 3)
        # nino data containing the target only
        self.target_nino = np.concatenate(target_nino, axis=0)  # (N, 24)
        print(sst1.shape)
        print(self.target_nino.shape)
        assert sst1.shape[0] == self.target_nino.shape[0]
        assert sst1.shape[1] == 38
        assert self.target_nino.shape[1] == 38

        if len(sst1) >= 1001:
            predData = np.load(r'/media/ruichuang/ENSO_predict/predictENSOData/CMIP5_pred13.npy')
            print("predData.shape=", predData.shape)
        elif len(sst1) <= 1000:
            predData = np.load(r'/media/ruichuang/ENSO_predict/predictENSOData/GODAS_pred.npy')
            print("predData.shape=", predData.shape)

        self.sst = np.concatenate((predData.transpose(0,1,3,4,2), sst1), axis=-1)

    def GetDataShape(self):
        return {'sst': self.sst.shape,
                'nino target': self.target_nino.shape}

    def __len__(self):
        return self.sst.shape[0]

    def __getitem__(self, idx):
        return self.sst[idx], self.target_nino[idx]
