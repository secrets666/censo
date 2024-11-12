from sa_convlstm import SAConvLSTM, SAConvLSTM_half, SAConvLSTM_9, SAConvLSTM_6, SAConvLSTM_3, SAConvLSTM_ensemble
import torch
import torch.nn as nn
from config import configs
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
from utils import *
import math


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model_zoo = [SAConvLSTM, SAConvLSTM_half, SAConvLSTM_3, SAConvLSTM_6, SAConvLSTM_9]

class Trainer:
    def __init__(self, configs):
        self.configs = configs
        self.device = configs.device
        torch.manual_seed(5)
        self.network = model_zoo[configs.model_i](configs).to(configs.device)

        # for name, param in self.network.named_parameters():
        #     print(name, param.shape, param.requires_grad)

        total_num = sum([param.nelement() for param in self.network.parameters()])
        print('Total params num: {}'.format(total_num))
        print('*****************Finish Parameter****************')

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=configs.lr, weight_decay=configs.weight_decay)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.3, patience=0, verbose=True, min_lr=0.0001)
        self.weight = torch.from_numpy(np.array([1.5]*4 + [2]*7 + [3]*7 + [4]*6) * np.log(np.arange(24)+1)).to(configs.device)

    def score(self, y_pred, y_true):
        with torch.no_grad():
            sc, cor = score(y_pred, y_true, self.weight)
        return sc.item(), cor

    def loss_sst(self, y_pred, y_true):
        # y_pred/y_true (N, 37, 24, 48)
        rmse = torch.mean((y_pred - y_true)**2, dim=[2, 3])
        rmse = torch.sum(rmse.sqrt().mean(dim=0))
        return rmse

    def loss_nino(self, y_pred, y_true):
        with torch.no_grad():
            rmse = torch.sqrt(torch.mean((y_pred - y_true)**2, dim=0)) * self.weight
        return rmse.sum()

    def train_once(self, sst, nino_true, ratio):
        '''use_hc=0 sst = (N, 38, 24, 48, 4), nino_true = (N, 38, 3)
           use_hc=1 sst = (N, 38, 24, 48, 4), nino_true = (N, 38, 3)'''
        sst_pred, nino_pred = self.network(sst.float(), nino_true, device=self.device, teacher_forcing=True,
                                           scheduled_sampling_ratio=ratio, train=True)
        self.optimizer.zero_grad()

        _, num_t, _, _, _ = sst_pred.size()
        loss_sst = self.loss_sst(sst_pred, sst[:, -num_t:, :, :, 0:2].permute(0, 1, 4, 2, 3).to(self.device))
        loss_nino = self.loss_nino(nino_pred, nino_true[:,12:36,0].float().to(self.device))
        loss_sst.backward()
        if configs.gradient_clipping:
            nn.utils.clip_grad_norm_(self.network.parameters(), configs.clipping_threshold)
        self.optimizer.step()
        return loss_sst.item(), loss_nino.item(), nino_pred

    def test(self, dataloader_test):
        nino_pred = []
        sst_pred = []
        with torch.no_grad():
            for sst, nino_true in dataloader_test:
                sst, nino = self.network(sst.float(), nino_true, device=self.device, train=True)
                nino_pred.append(nino)
                sst_pred.append(sst)
        return torch.cat(sst_pred, dim=0), torch.cat(nino_pred, dim=0)

    def infer(self, dataset, dataloader, save_data=False):
        # calculate loss_func and score on a eval/test set
        self.network.eval()
        with torch.no_grad():
            sst_pred, nino_pred = self.test(dataloader)  # y_pred for the whole evalset
            nino_true = torch.from_numpy(dataset.target_nino[:,12:36,0]).float().to(self.device)
            sst_true = torch.from_numpy(dataset.sst[:, :, :, :, 0:2]).permute(0, 1, 4, 2, 3).float().to(self.device)
            sc, cor = self.score(nino_pred, nino_true)
            sst_pred = torch.cat((sst_true[:, :1, :, :, :], sst_pred), dim=1)
            assert sst_pred.shape[1] == 38
            # loss_sst = self.loss_sst(sst_pred, sst_true).item()
            # loss_nino = self.loss_nino(nino_pred, nino_true).item()
            loss_sst = 0
            loss_nino = 0


            if save_data:
                sst_pred = sst_pred.to("cpu").numpy()
                print("sst_pred.shape=", sst_pred.shape)
                np.save('/media/ruichuang/ENSO_predict/predictENSOData/CMIP5_pred13.npy', sst_pred)
                sst_true = sst_true.to("cpu").numpy()
                print("sst_true.shape=", sst_true.shape)
                np.save('/media/ruichuang/ENSO_predict/predictENSOData//CMIP5_true13.npy', sst_true)

                # nino_pred = nino_pred.to("cpu").numpy()
                # print("nino_pred.shape=", nino_pred.shape)
                # np.save('../../best_model_data/hybrid/godas/nino_pred.npy', nino_pred)
                # nino_true = nino_true.to("cpu").numpy()
                # print("nino_true.shape=", nino_true.shape)
                # np.save('../../best_model_data/hybrid/godas/nino_true.npy', nino_true)

        return loss_sst, loss_nino, sc, cor

    def train(self, dataset_train, dataset_eval, chk_path):
        torch.manual_seed(0)
        print('loading train dataloader')
        dataloader_train = DataLoader(dataset_train, batch_size=self.configs.batch_size, shuffle=False)
        print('loading eval dataloader')
        dataloader_eval = DataLoader(dataset_eval, batch_size=self.configs.batch_size_test, shuffle=False)

        count = 0
        best = - math.inf
        ssr_ratio = 1
        for i in range(self.configs.num_epochs):
            print('\nepoch: {0}'.format(i+1))
            self.network.train()

            for j, (sst, nino_true) in enumerate(dataloader_train):
                '''
                sst=(N,38,24,48,4), nino_true=(N,24,3)
                '''
                if ssr_ratio > 0:
                    ssr_ratio = max(ssr_ratio - self.configs.ssr_decay_rate, 0)
                loss_sst, loss_nino, nino_pred = self.train_once(sst, nino_true, ssr_ratio)

                if j % self.configs.display_interval == 0:
                    sc, _ = self.score(nino_pred, nino_true[:,12:36,0].float().to(self.device))
                    print('batch training loss: {:.2f}, {:.2f}, score: {:.4f}, ssr ratio: {:.4f}'.format(loss_sst, loss_nino, sc, ssr_ratio))
                    # loss_sst_eval, loss_nino_eval, sc_eval, cor = self.infer(dataset=dataset_eval,
                    #                                                          dataloader=dataloader_eval)
                    # print(cor)

            # evaluation
            loss_sst_eval, loss_nino_eval, sc_eval, cor = self.infer(dataset=dataset_eval, dataloader=dataloader_eval)
            print(cor)
            print('epoch eval loss:\nsst: {:.2f}, nino: {:.2f}, sc: {:.4f}'.format(loss_sst_eval, loss_nino_eval, sc_eval))
            self.lr_scheduler.step(sc_eval)
            if sc_eval <= best:
                count += 1
                print('eval score is not improved for {} epoch'.format(count))
            elif self.configs.save_last == 1 and i == self.configs.num_epoch-1:
                print('save last epoch')
                self.save_model(chk_path)
                best = sc_eval
            else:
                count = 0
                print('eval score is improved from {:.5f} to {:.5f}, saving model'.format(best, sc_eval))
                self.save_model(chk_path)
                best = sc_eval

            if count == self.configs.patience:
                print('early stopping reached, best score is {:5f}'.format(best))
                break

    def save_configs(self, config_path):
        with open(config_path, 'wb') as path:
            pickle.dump(self.configs, path)

    def save_model(self, path):
        torch.save({'net': self.network.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, path)


def prepare_data(ds_dir, configs):
    # train/eval/test split
    cmip6sst, cmip6nino = read_raw_data(ds_dir, 'CMIP', configs)  # cmip6sst=(1692,24,48,21,3) where (:,:,:,:,0)sst (:,:,:,:,1:3)month_embed   cmip6nino=(1692,21, 3)
    sodasst, sodanino = read_raw_data(ds_dir, 'SODA', configs)  # (1200,24,48,1,3) (:,:,:,:,0)sst  (:,:,:,:,1:3)month_embed  (1200,1)
    godassst, godasnino = read_raw_data(ds_dir, 'GODAS', configs)  # (432,24,48,1,3) (:,:,:,:,0)sst  (:,:,:,:,1:3)month_embed  (432,1)

    sst_train = cmip6sst
    nino_train = cmip6nino
    sst_eval = godassst
    nino_eval =godasnino
    sst_test = godassst
    nino_test = godasnino
    return sst_train, nino_train, sst_eval, nino_eval, sst_test, nino_test


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='test')

    parser.add_argument('--hidden_dim_1', '-hd1', default=(128, 128, 128, 128), type=lambda x: tuple(map(int, x.split(','))), help='hiddem dim')
    parser.add_argument('--hidden_dim_2', '-hd2', default=(32, 32, 32, 32), type=lambda x: tuple(map(int, x.split(','))), help='hiddem dim')
    parser.add_argument('--hidden_dim_3', '-hd3', default=(32, 32, 32, 32), type=lambda x: tuple(map(int, x.split(','))), help='hiddem dim')
    parser.add_argument('--hidden_dim_4', '-hd4', default=(32, 32, 32, 32), type=lambda x: tuple(map(int, x.split(','))), help='hiddem dim')
    parser.add_argument('--hidden_dim_5', '-hd5', default=(32, 32, 32, 32), type=lambda x: tuple(map(int, x.split(','))), help='hiddem dim')

    parser.add_argument('--d_attn_1', '-da1', default=64, type=int, help='d attn')
    parser.add_argument('--d_attn_2', '-da2', default=16, type=int, help='d attn')
    parser.add_argument('--d_attn_3', '-da3', default=16, type=int, help='d attn')
    parser.add_argument('--d_attn_4', '-da4', default=16, type=int, help='d attn')
    parser.add_argument('--d_attn_5', '-da5', default=16, type=int, help='d attn')

    parser.add_argument('--model_i', '-mi', default=0, type=int, help='model i')
    parser.add_argument('--gpu', '-g', default=0, type=int, help='gpu')
    parser.add_argument('--use_hc', '-hc', default=1, type=int, help='gpu')
    parser.add_argument('--time', '-t', default='hybrid', type=str, help='no, stable, learn, hybrid')
    parser.add_argument('--path', '-p', default='./', type=str, help='')
    parser.add_argument('--epoch', '-e', default=5, type=int, help='')
    parser.add_argument('--save_last', '-s', default=0, type=int, help='save given epoch model')

    args = parser.parse_args()

    configs.hidden_dim_1 = args.hidden_dim_1
    configs.hidden_dim_2 = args.hidden_dim_2
    configs.hidden_dim_3 = args.hidden_dim_3
    configs.hidden_dim_4 = args.hidden_dim_4
    configs.hidden_dim_5 = args.hidden_dim_5

    configs.d_attn_1 = args.d_attn_1
    configs.d_attn_2 = args.d_attn_2
    configs.d_attn_3 = args.d_attn_3
    configs.d_attn_4 = args.d_attn_4
    configs.d_attn_5 = args.d_attn_5

    configs.model_i = args.model_i
    configs.device = torch.device('cuda:{}'.format(args.gpu))
    configs.use_hc = args.use_hc
    configs.time = args.time
    configs.num_epoch = args.epoch
    configs.save_last = args.save_last


    if args.time == 'no':
        pass  # 默认为2
    elif args.time == 'stable' or args.time == 'learn':
        configs.input_dim = 4
    elif args.time == 'hybrid':
        configs.input_dim = 6


    print(configs.__dict__)

    print('\nreading data')
    # sst_train, nino_train, sst_eval, nino_eval, sst_test, nino_test = prepare_data(r'E:\tianchi\data_all\enso2tc_month\\', configs)
    sst_train, nino_train, sst_eval, nino_eval, sst_test, nino_test = prepare_data(r'/media/ruichuang/ENSO_predict/enso2tc_add_mon/', configs)
    print(sst_train.shape, sst_eval.shape)
    print(nino_train.shape, nino_eval.shape)

    print('processing training set')
    dataset_train = cmip_dataset(sst_train, nino_train, samples_gap=13, start=0)
    print(dataset_train.GetDataShape())  # {'sst':(6951,38,24,48,3), 'nino_target':(6951,24)}
    del sst_train
    del nino_train
    print('processing eval set')
    dataset_eval = cmip_dataset(sst_eval, nino_eval, samples_gap=1)
    print(dataset_eval.GetDataShape())  # {'sst':(395,38,24,48,3), 'nino_target':(395,24)}
    del sst_eval
    del nino_eval


    trainer = Trainer(configs)
    trainer.save_configs(args.path + '/config_train.pkl')
    # trainer.train(dataset_train, dataset_eval, args.path + '/checkpoint.chk')
    print('\n----- training finished -----\n')

    # del dataset_train
    del dataset_eval

    print('processing test set')
    dataset_test = cmip_dataset(sst_test, nino_test, samples_gap=1)
    print(dataset_test.GetDataShape())

    # np.save('../../best_model_data/godas/nino_true.npy', dataset_test.target_nino)
    # quit()

    # test
    print('loading test dataloader')
    dataloader_test = DataLoader(dataset_train, batch_size=configs.batch_size_test, shuffle=False)
    chk = torch.load('checkpoint.chk', map_location=configs.device)
    trainer.network.load_state_dict(chk['net'])
    print('testing...')
    loss_sst_test, loss_nino_test, sc_test, cor = trainer.infer(dataset=dataset_train, dataloader=dataloader_test, save_data=True)
    print(cor)
    print('test loss:\n sst: {:.2f}, nino: {:.2f}, score: {:.4f}'.format(loss_sst_test, loss_nino_test, sc_test))




