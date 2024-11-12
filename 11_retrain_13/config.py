import torch


class Configs:
    def __init__(self):
        pass


configs = Configs()

# trainer related
configs.n_cpu = 0
configs.device = torch.device('cuda:0')
configs.batch_size_test = 40
configs.batch_size = 2
configs.lr = 0.001
configs.weight_decay = 0
configs.display_interval = 250
configs.num_epochs = 50
configs.early_stopping = True
configs.patience = 10
configs.gradient_clipping = True
configs.clipping_threshold = 1.

# data related
configs.input_dim = 2
configs.output_dim = 2
configs.input_length = 12
configs.output_length = 24
configs.input_gap = 1
configs.pred_shift = 24

# model i
configs.model_i = 1

# model1 related
configs.kernel_size = (3, 3)
configs.bias = True
configs.hidden_dim_1 = (16, 16, 16, 16)
configs.d_attn_1 = 2
configs.ssr_decay_rate = 0.8e-4


# model2 related
configs.hidden_dim_2 = (16, 16, 16, 16)
configs.d_attn_2 = 2


# model3 related
configs.hidden_dim_3 = (16, 16, 16, 16)
configs.d_attn_3 = 2


# model4 related
configs.hidden_dim_4 = (16, 16, 16, 16)
configs.d_attn_4 = 2


# model5 related
configs.hidden_dim_5 = (16, 16, 16, 16)
configs.d_attn_5 = 2

configs.use_hc = 0
configs.time = 'learn'
configs.save_last = 0


