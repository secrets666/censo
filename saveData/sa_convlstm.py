"""
Author: written by Jiacheng WU
The model architecture is adopted from SA-ConvLSTM (Lin et al., 2020) 
(https://ojs.aaai.org/index.php/AAAI/article/view/6819)
"""

import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def attn(query, key, value):
    """
    Apply attention over the spatial dimension (S)
    Args:
        query, key, value: (N, C, S)
    Returns:
        output of the same size
    """
    scores = query.transpose(1, 2) @ key / math.sqrt(query.size(1))  # (N, S, S)
    attn = F.softmax(scores, dim=-1)
    output = attn @ value.transpose(1, 2)
    return output.transpose(1, 2)  # (N, C, S)


class SAAttnMem(nn.Module):
    def __init__(self, input_dim, d_model, kernel_size):
        """
        The self-attention memory module added to ConvLSTM
        """
        super().__init__()
        pad = kernel_size[0] // 2, kernel_size[1] // 2
        self.d_model = d_model
        self.input_dim = input_dim
        self.conv_h = nn.Conv2d(input_dim, d_model*3, kernel_size=1)
        self.conv_m = nn.Conv2d(input_dim, d_model*2, kernel_size=1)
        self.conv_z = nn.Conv2d(d_model*2, d_model, kernel_size=1)
        self.conv_output = nn.Conv2d(input_dim+d_model, input_dim*3, kernel_size=kernel_size, padding=pad)

    def forward(self, h, m):
        hq, hk, hv = torch.split(self.conv_h(h), self.d_model, dim=1)
        mk, mv = torch.split(self.conv_m(m), self.d_model, dim=1)
        N, C, H, W = hq.size()
        Zh = attn(hq.view(N, C, -1), hk.view(N, C, -1), hv.view(N, C, -1))  # (N, S, C)
        Zm = attn(hq.view(N, C, -1), mk.view(N, C, -1), mv.view(N, C, -1))  # (N, S, C)
        Z = self.conv_z(torch.cat([Zh.view(N, C, H, W), Zm.view(N, C, H, W)], dim=1))
        i, g, o = torch.split(self.conv_output(torch.cat([Z, h], dim=1)), self.input_dim, dim=1)
        i = torch.sigmoid(i)
        g = torch.tanh(g)
        m_next = i * g + (1 - i) * m
        h_next = torch.sigmoid(o) * m_next
        return h_next, m_next


class SAConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, d_attn, kernel_size):
        """
        The SA-ConvLSTM cell module. Same as the ConvLSTM cell except with the
        self-attention memory module and the M added
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        pad = kernel_size[0] // 2, kernel_size[1] // 2

        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=pad)
        self.sa = SAAttnMem(input_dim=hidden_dim, d_model=d_attn, kernel_size=kernel_size)

    def initialize(self, inputs):
        device = inputs.device
        batch_size, _, height, width = inputs.size()

        self.hidden_state = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        self.cell_state = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        self.memory_state = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)

    def forward(self, inputs, first_step=False):
        if first_step:
            self.initialize(inputs)

        combined = torch.cat([inputs, self.hidden_state], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        self.cell_state = f * self.cell_state + i * g
        self.hidden_state = o * torch.tanh(self.cell_state)
        # novel for sa-convlstm
        self.hidden_state, self.memory_state = self.sa(self.hidden_state, self.memory_state)
        return self.hidden_state


class SAConvLSTM(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        input_dim = configs.input_dim
        hidden_dim = configs.hidden_dim_1
        d_attn = configs.d_attn_1
        kernel_size = configs.kernel_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = len(hidden_dim)

        layers = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            layers.append(SAConvLSTMCell(input_dim=cur_input_dim,
                                         hidden_dim=self.hidden_dim[i],
                                         d_attn=d_attn,
                                         kernel_size=kernel_size))

        self.layers = nn.ModuleList(layers)
        self.conv_output = nn.Conv2d(self.hidden_dim[-1], configs.output_dim, kernel_size=1)

        if self.configs.time == 'learn' or self.configs.time == 'hybrid':
            self.mon_embed = nn.Embedding(12, 24*48)
            self.season_embed = nn.Embedding(4, 24*48)

    def forward(self, input_x, input_y=None, device=torch.device('cuda:0'), input_frames=12, future_frames=26, output_frames=37,
                teacher_forcing=False, scheduled_sampling_ratio=0, train=True):
        """
        The self-attention ConvLSTM module, employed with scheduled sampling
        for multi-step spatio-temporalforecasting.
        The network is designed to predict the next frame based on the context in the current time step,
        and multi-step forecasts are made by recursively invoking the SAConvLSTMCell.
        The sst in the input time period are also used as the ground truth for training
        Args:
            input_x: input with size (N, T, C, H, W)
            input_frames: the number of input time steps
            future_frames: the number of target time steps for SST
            output_frames: the number of model output time steps, typically equal to
                           input_frames + future_frames - 1 (training) or future_frames (testing)
            teacher_forcing: specify if the teacher forcing is used. Expect True (training), False (testing)
            scheduled_sampling_ratio: The sampling ratio used during scheduled sampling 
            train: specify whether or not the model is in the train mode
        Returns:
            outputs: the predicted SST with size (N, output_frames, H, W) for backward propagation
            nino_pred: the predicted nino with size (N, future_frames)
        """
        '''input_x=(N,38,24,48,3) input_y=(N,38,3)'''
        assert len(input_x.shape) == 5
        input_x = input_x.to(device)
        input_y = input_y.to(device)
        bsz, num_t, h, w, _ = input_x.size()
        if self.configs.time == 'learn' or self.configs.time == 'hybrid':
            # get mon and season embed
            mon = input_y[:,:,1].to(torch.int)
            season = input_y[:,:,2].to(torch.int)
            mon_embed = self.mon_embed(mon).reshape(bsz,num_t,h,w)  # (2,38,24,48)
            season_embed = self.season_embed(season).reshape(bsz,num_t,h,w)  # (2,38,24,48)

            if self.configs.time == 'learn':
                input_x[:,:,:,:,2] = mon_embed
                input_x[:,:,:,:,3] = season_embed
            elif self.configs.time == 'hybrid':
                input_x = torch.cat([input_x, mon_embed[:,:,:,:,None], season_embed[:,:,:,:,None]], dim=-1)
        elif self.configs.time == 'no':
            input_x = input_x[:,:,:,:,:2]
        elif self.configs.time == 'stable':
            input_x = input_x

        input_x = input_x.permute(0,1,4,2,3)  # (N,38,3,24,48)
        if train:
            if teacher_forcing and scheduled_sampling_ratio > 1e-6:
                teacher_forcing_mask = torch.bernoulli(scheduled_sampling_ratio *
                    torch.ones(input_x.size(0), future_frames - 1, 1, 1, 1))
            else:
                teacher_forcing = False
        else:
            # assert input_x.size(1) == input_frames
            teacher_forcing = False

        total_steps = input_frames + future_frames - 1
        outputs = [None] * total_steps

        for t in range(total_steps):
            if t < input_frames:
                input_ = input_x[:, t].to(device)
            elif not teacher_forcing:
                input_ = torch.cat((outputs[t - 1], input_x[:, t, 2:, :, :].to(device)), dim=1)
            else:
                mask = teacher_forcing_mask[:, t - input_frames].float().to(device)
                input_ = input_x[:, t, :2].to(device) * mask + outputs[t - 1] * (1 - mask)
                input_ = torch.cat((input_, input_x[:, t, 2:].to(device)), dim=1)

            first_step = (t == 0)
            input_ = input_.float()

            for layer_idx in range(self.num_layers):
                input_ = self.layers[layer_idx](input_, first_step=first_step)

            if train or (t >= (input_frames - 1)):
                outputs[t] = self.conv_output(input_)

        outputs = [x for x in outputs if x is not None]
        if train:
            assert len(outputs) == output_frames
        else:
            assert len(outputs) == future_frames

        outputs = torch.stack(outputs, dim=1) # (N, 37, H, W)
        nino_pred = outputs[:, -future_frames:, 0, 10:13, 19:30].mean(dim=[2, 3])  # (N, 26)
        nino_pred = nino_pred.unfold(dimension=1, size=3, step=1).mean(dim=2)  # (N, 24)
        return outputs, nino_pred
        # rmse = torch.sqrt((nino_pred - input_y[:,12:36,0]) ** 2)
        # return rmse
    # (N,37,2,H,W) (N,24)
    # (N,26,2,H,W) (N,24)


class SAConvLSTM_half(nn.Module):
    def __init__(self, configs):
        super().__init__()
        input_dim = configs.input_dim
        hidden_dim = configs.hidden_dim_2
        d_attn = configs.d_attn_2
        kernel_size = configs.kernel_size

        self.input_dim = (input_dim - 2) * 4 + 2  # 6
        self.hidden_dim = hidden_dim
        self.num_layers = len(hidden_dim)
        layers = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            layers.append(SAConvLSTMCell(input_dim=cur_input_dim,
                                         hidden_dim=self.hidden_dim[i],
                                         d_attn=d_attn,
                                         kernel_size=kernel_size))

        self.layers = nn.ModuleList(layers)
        self.conv_output = nn.Conv2d(self.hidden_dim[-1], 4, kernel_size=1)

    def forward(self, input_x, device=torch.device('cuda:0'), input_frames=12, future_frames=26, output_frames=37,
                teacher_forcing=False, scheduled_sampling_ratio=0, train=True):
        """
        The self-attention ConvLSTM module, employed with scheduled sampling
        for multi-step spatio-temporalforecasting.
        The network is designed to predict the next frame based on the context in the current time step,
        and multi-step forecasts are made by recursively invoking the SAConvLSTMCell.
        The sst in the input time period are also used as the ground truth for training
        Args:
            input_x: input with size (N, T, C, H, W)
            input_frames: the number of input time steps
            future_frames: the number of target time steps for SST
            output_frames: the number of model output time steps, typically equal to
                           input_frames + future_frames - 1 (training) or future_frames (testing)
            teacher_forcing: specify if the teacher forcing is used. Expect True (training), False (testing)
            scheduled_sampling_ratio: The sampling ratio used during scheduled sampling
            train: specify whether or not the model is in the train mode
        Returns:
            outputs: the predicted SST with size (N, output_frames, H, W) for backward propagation
            nino_pred: the predicted nino with size (N, future_frames)
        """
        '''input_x=(N,38,24,48,3)'''
        _,_,h,w,_ = input_x.size()
        assert len(input_x.shape) == 5
        # down-sampling
        half_input_x = []
        for i in range(2):
            for j in range(2):
                half_input_x.append(input_x[:,:,i::2,j::2,0])

        half_input_x = torch.stack(half_input_x, dim=2)
        half_input_x = torch.cat([half_input_x, input_x[:,:,::2,::2, 1][:,:,None,:,:], input_x[:,:,::2,::2, 2][:,:,None,:,:]], dim=2)
        # print(half_input_x.size())  # (2,38,6,12,24)

        input_x = input_x.permute(0,1,4,2,3)  # (N,38,3,24,48)
        if train:
            if teacher_forcing and scheduled_sampling_ratio > 1e-6:
                teacher_forcing_mask = torch.bernoulli(scheduled_sampling_ratio *
                    torch.ones(half_input_x.size(0), future_frames - 1, 1, 1, 1))
            else:
                teacher_forcing = False
        else:
            # assert input_x.size(1) == input_frames
            teacher_forcing = False

        total_steps = input_frames + future_frames - 1
        outputs = [None] * total_steps

        for t in range(total_steps):
            if t < input_frames:
                input_ = half_input_x[:, t].to(device)
            elif not teacher_forcing:
                input_ = torch.cat((outputs[t-1], half_input_x[:,t,4:,:,:].to(device)), dim=1)
            else:
                mask = teacher_forcing_mask[:, t - input_frames].float().to(device)
                input_ = half_input_x[:, t, :4].to(device) * mask + outputs[t-1] * (1 - mask)
                input_ = torch.cat((input_, half_input_x[:,t,4:].to(device)), dim=1)

            first_step = (t == 0)
            input_ = input_.float()

            for layer_idx in range(self.num_layers):
                input_ = self.layers[layer_idx](input_, first_step=first_step)

            if train or (t >= (input_frames - 1)):
                outputs[t] = self.conv_output(input_)
                # print(outputs[t].size())  # (2,4,12,24)


        outputs = [x for x in outputs if x is not None]
        if train:
            assert len(outputs) == output_frames
        else:
            assert len(outputs) == future_frames


        outputs = torch.stack(outputs, dim=1)  # (N, 37, 4, H, W)
        bsz, num_t, _, _, _ = outputs.size()
        outputs1 = torch.empty(bsz,num_t,h,w).to(device)
        k = 0
        for i in range(2):
            for j in range(2):
                outputs1[:,:,i::2,j::2] = outputs[:,:,k,:,:]
                k += 1

        nino_pred = outputs1[:, -future_frames:, 10:13, 19:30].mean(dim=[2, 3])  # (N, 26)
        nino_pred = nino_pred.unfold(dimension=1, size=3, step=1).mean(dim=2)  # (N, 24)
        return outputs1, nino_pred


class SAConvLSTM_3(nn.Module):
    def __init__(self, configs):
        super().__init__()
        input_dim = configs.input_dim
        hidden_dim = configs.hidden_dim_3
        d_attn = configs.d_attn_3
        kernel_size = configs.kernel_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = len(hidden_dim)

        layers = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            layers.append(SAConvLSTMCell(input_dim=cur_input_dim,
                                         hidden_dim=self.hidden_dim[i],
                                         d_attn=d_attn,
                                         kernel_size=kernel_size))

        self.layers = nn.ModuleList(layers)
        self.conv_output = nn.Conv2d(self.hidden_dim[-1], 1, kernel_size=1)

    def forward(self, input_x, device=torch.device('cuda:0'), input_frames=12, future_frames=26, output_frames=37,
                teacher_forcing=False, scheduled_sampling_ratio=0, train=True):
        """
        The self-attention ConvLSTM module, employed with scheduled sampling
        for multi-step spatio-temporalforecasting.
        The network is designed to predict the next frame based on the context in the current time step,
        and multi-step forecasts are made by recursively invoking the SAConvLSTMCell.
        The sst in the input time period are also used as the ground truth for training
        Args:
            input_x: input with size (N, T, C, H, W)
            input_frames: the number of input time steps
            future_frames: the number of target time steps for SST
            output_frames: the number of model output time steps, typically equal to
                           input_frames + future_frames - 1 (training) or future_frames (testing)
            teacher_forcing: specify if the teacher forcing is used. Expect True (training), False (testing)
            scheduled_sampling_ratio: The sampling ratio used during scheduled sampling
            train: specify whether or not the model is in the train mode
        Returns:
            outputs: the predicted SST with size (N, output_frames, H, W) for backward propagation
            nino_pred: the predicted nino with size (N, future_frames)
        """
        '''input_x=(N,38,24,48,3)'''
        assert len(input_x.shape) == 5
        input_x = input_x.permute(0,1,4,2,3)  # (N,38,3,24,48)
        if train:
            if teacher_forcing and scheduled_sampling_ratio > 1e-6:
                teacher_forcing_mask = torch.bernoulli(scheduled_sampling_ratio *
                    torch.ones(input_x.size(0), future_frames - 1, 1, 1, 1))
            else:
                teacher_forcing = False
        else:
            # assert input_x.size(1) == input_frames
            teacher_forcing = False

        total_steps = input_frames + future_frames - 1
        outputs = [None] * total_steps

        for t in range(9, total_steps):
            if t < input_frames:
                input_ = input_x[:, t].to(device)
            elif not teacher_forcing:
                input_ = torch.cat((outputs[t-1], input_x[:,t,1:,:,:].to(device)), dim=1)
            else:
                mask = teacher_forcing_mask[:, t - input_frames].float().to(device)
                input_ = input_x[:, t, :1].to(device) * mask + outputs[t-1] * (1 - mask)
                input_ = torch.cat((input_, input_x[:,t,1:].to(device)), dim=1)

            first_step = (t == 9)
            input_ = input_.float()

            for layer_idx in range(self.num_layers):
                input_ = self.layers[layer_idx](input_, first_step=first_step)

            if train or (t >= (input_frames - 1)):
                outputs[t] = self.conv_output(input_)

        outputs = [x for x in outputs if x is not None]
        if train:
            assert len(outputs) == output_frames - 9
        else:
            assert len(outputs) == future_frames

        outputs = torch.stack(outputs, dim=1)[:, :, 0]  # (N, 37-9, H, W)
        nino_pred = outputs[:, -future_frames:, 10:13, 19:30].mean(dim=[2, 3])  # (N, 26)
        nino_pred = nino_pred.unfold(dimension=1, size=3, step=1).mean(dim=2)  # (N, 24)
        return outputs, nino_pred
    # torch.Size([2, 28, 24, 48])
    # torch.Size([2, 24])


class SAConvLSTM_6(nn.Module):
    def __init__(self, configs):
        super().__init__()
        input_dim = configs.input_dim
        hidden_dim = configs.hidden_dim_4
        d_attn = configs.d_attn_4
        kernel_size = configs.kernel_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = len(hidden_dim)

        layers = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            layers.append(SAConvLSTMCell(input_dim=cur_input_dim,
                                         hidden_dim=self.hidden_dim[i],
                                         d_attn=d_attn,
                                         kernel_size=kernel_size))

        self.layers = nn.ModuleList(layers)
        self.conv_output = nn.Conv2d(self.hidden_dim[-1], 1, kernel_size=1)

    def forward(self, input_x, device=torch.device('cuda:0'), input_frames=12, future_frames=26, output_frames=37,
                teacher_forcing=False, scheduled_sampling_ratio=0, train=True):
        """
        The self-attention ConvLSTM module, employed with scheduled sampling
        for multi-step spatio-temporalforecasting.
        The network is designed to predict the next frame based on the context in the current time step,
        and multi-step forecasts are made by recursively invoking the SAConvLSTMCell.
        The sst in the input time period are also used as the ground truth for training
        Args:
            input_x: input with size (N, T, C, H, W)
            input_frames: the number of input time steps
            future_frames: the number of target time steps for SST
            output_frames: the number of model output time steps, typically equal to
                           input_frames + future_frames - 1 (training) or future_frames (testing)
            teacher_forcing: specify if the teacher forcing is used. Expect True (training), False (testing)
            scheduled_sampling_ratio: The sampling ratio used during scheduled sampling
            train: specify whether or not the model is in the train mode
        Returns:
            outputs: the predicted SST with size (N, output_frames, H, W) for backward propagation
            nino_pred: the predicted nino with size (N, future_frames)
        """
        '''input_x=(N,38,24,48,3)'''
        assert len(input_x.shape) == 5
        input_x = input_x.permute(0,1,4,2,3)  # (N,38,3,24,48)
        if train:
            if teacher_forcing and scheduled_sampling_ratio > 1e-6:
                teacher_forcing_mask = torch.bernoulli(scheduled_sampling_ratio *
                    torch.ones(input_x.size(0), future_frames - 1, 1, 1, 1))
            else:
                teacher_forcing = False
        else:
            # assert input_x.size(1) == input_frames
            teacher_forcing = False

        total_steps = input_frames + future_frames - 1
        outputs = [None] * total_steps

        for t in range(6, total_steps):
            if t < input_frames:
                input_ = input_x[:, t].to(device)
            elif not teacher_forcing:
                input_ = torch.cat((outputs[t-1], input_x[:,t,1:,:,:].to(device)), dim=1)
            else:
                mask = teacher_forcing_mask[:, t - input_frames].float().to(device)
                input_ = input_x[:, t, :1].to(device) * mask + outputs[t-1] * (1 - mask)
                input_ = torch.cat((input_, input_x[:,t,1:].to(device)), dim=1)

            first_step = (t == 6)
            input_ = input_.float()

            for layer_idx in range(self.num_layers):
                input_ = self.layers[layer_idx](input_, first_step=first_step)

            if train or (t >= (input_frames - 1)):
                outputs[t] = self.conv_output(input_)

        outputs = [x for x in outputs if x is not None]
        if train:
            assert len(outputs) == output_frames - 6
        else:
            assert len(outputs) == future_frames

        outputs = torch.stack(outputs, dim=1)[:, :, 0]  # (N, 37-6, H, W)
        nino_pred = outputs[:, -future_frames:, 10:13, 19:30].mean(dim=[2, 3])  # (N, 26)
        nino_pred = nino_pred.unfold(dimension=1, size=3, step=1).mean(dim=2)  # (N, 24)
        return outputs, nino_pred
    # torch.Size([2, 31, 24, 48])
    # torch.Size([2, 24])


class SAConvLSTM_9(nn.Module):
    def __init__(self, configs):
        super().__init__()
        input_dim = configs.input_dim
        hidden_dim = configs.hidden_dim_5
        d_attn = configs.d_attn_5
        kernel_size = configs.kernel_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = len(hidden_dim)

        layers = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            layers.append(SAConvLSTMCell(input_dim=cur_input_dim,
                                         hidden_dim=self.hidden_dim[i],
                                         d_attn=d_attn,
                                         kernel_size=kernel_size))

        self.layers = nn.ModuleList(layers)
        self.conv_output = nn.Conv2d(self.hidden_dim[-1], 1, kernel_size=1)

    def forward(self, input_x, device=torch.device('cuda:0'), input_frames=12, future_frames=26, output_frames=37,
                teacher_forcing=False, scheduled_sampling_ratio=0, train=True):
        """
        The self-attention ConvLSTM module, employed with scheduled sampling
        for multi-step spatio-temporalforecasting.
        The network is designed to predict the next frame based on the context in the current time step,
        and multi-step forecasts are made by recursively invoking the SAConvLSTMCell.
        The sst in the input time period are also used as the ground truth for training
        Args:
            input_x: input with size (N, T, C, H, W)
            input_frames: the number of input time steps
            future_frames: the number of target time steps for SST
            output_frames: the number of model output time steps, typically equal to
                           input_frames + future_frames - 1 (training) or future_frames (testing)
            teacher_forcing: specify if the teacher forcing is used. Expect True (training), False (testing)
            scheduled_sampling_ratio: The sampling ratio used during scheduled sampling
            train: specify whether or not the model is in the train mode
        Returns:
            outputs: the predicted SST with size (N, output_frames, H, W) for backward propagation
            nino_pred: the predicted nino with size (N, future_frames)
        """
        '''input_x=(N,38,24,48,3)'''
        assert len(input_x.shape) == 5
        input_x = input_x.permute(0,1,4,2,3)  # (N,38,3,24,48)
        if train:
            if teacher_forcing and scheduled_sampling_ratio > 1e-6:
                teacher_forcing_mask = torch.bernoulli(scheduled_sampling_ratio *
                    torch.ones(input_x.size(0), future_frames - 1, 1, 1, 1))
            else:
                teacher_forcing = False
        else:
            # assert input_x.size(1) == input_frames
            teacher_forcing = False

        total_steps = input_frames + future_frames - 1
        outputs = [None] * total_steps

        for t in range(3, total_steps):
            if t < input_frames:
                input_ = input_x[:, t].to(device)
            elif not teacher_forcing:
                input_ = torch.cat((outputs[t-1], input_x[:,t,1:,:,:].to(device)), dim=1)
            else:
                mask = teacher_forcing_mask[:, t - input_frames].float().to(device)
                input_ = input_x[:, t, :1].to(device) * mask + outputs[t-1] * (1 - mask)
                input_ = torch.cat((input_, input_x[:,t,1:].to(device)), dim=1)

            first_step = (t == 3)
            input_ = input_.float()

            for layer_idx in range(self.num_layers):
                input_ = self.layers[layer_idx](input_, first_step=first_step)

            if train or (t >= (input_frames - 1)):
                outputs[t] = self.conv_output(input_)

        outputs = [x for x in outputs if x is not None]
        if train:
            assert len(outputs) == output_frames - 3
        else:
            assert len(outputs) == future_frames

        outputs = torch.stack(outputs, dim=1)[:, :, 0]  # (N, 37-9, H, W)
        nino_pred = outputs[:, -future_frames:, 10:13, 19:30].mean(dim=[2, 3])  # (N, 26)
        nino_pred = nino_pred.unfold(dimension=1, size=3, step=1).mean(dim=2)  # (N, 24)
        return outputs, nino_pred
    # torch.Size([2, 34, 24, 48])
    # torch.Size([2, 24])


class SAConvLSTM_ensemble(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.sa = SAConvLSTM(configs)
        self.sa_half = SAConvLSTM_half(configs)
        self.sa_3 = SAConvLSTM_3(configs)
        self.sa_6 = SAConvLSTM_6(configs)
        self.sa_9 = SAConvLSTM_9(configs)

    def forward(self, input_x, device=torch.device('cuda:0'), input_frames=12, future_frames=26, output_frames=37,
                teacher_forcing=False, scheduled_sampling_ratio=0, train=True):
        sst_p, nino_p = self.sa(input_x, device=device, input_frames=input_frames, future_frames=future_frames, output_frames=output_frames,
                                teacher_forcing=teacher_forcing, scheduled_sampling_ratio=scheduled_sampling_ratio, train=train)
        # (bsz,37,24,48) (bsz,24)

        sst_p_half, nino_p_half = self.sa_half(input_x, device=device, input_frames=input_frames, future_frames=future_frames, output_frames=output_frames,
                                               teacher_forcing=teacher_forcing, scheduled_sampling_ratio=scheduled_sampling_ratio, train=train)
        # (bsz,37,24,48) (bsz,24)
        sst_p_all = (sst_p + sst_p_half) / 2

        sst_p_3, nino_p_3 = self.sa_3(input_x, device=device, input_frames=input_frames, future_frames=future_frames, output_frames=output_frames,
                                      teacher_forcing=teacher_forcing, scheduled_sampling_ratio=scheduled_sampling_ratio, train=train)
        # (bsz,28,24,48) (bsz,24)
        if train:
            sst_p_3 = torch.cat([sst_p_all[:,:9,:,:], sst_p_3], dim=1)
        assert sst_p_3.size() == sst_p_all.size()
        sst_p_all = (sst_p_all + sst_p_3) / 2

        sst_p_6, nino_p_6 = self.sa_6(input_x, device=device, input_frames=input_frames, future_frames=future_frames, output_frames=output_frames,
                                      teacher_forcing=teacher_forcing, scheduled_sampling_ratio=scheduled_sampling_ratio, train=train)
        # (bsz,31,24,48) (bsz,24)
        if train:
            sst_p_6 = torch.cat([sst_p_all[:,:6,:,:], sst_p_6], dim=1)
        assert sst_p_6.size() == sst_p_all.size()
        sst_p_all = (sst_p_all + sst_p_6) / 2

        sst_p_9, nino_p_9 = self.sa_9(input_x, device=device, input_frames=input_frames, future_frames=future_frames, output_frames=output_frames,
                                      teacher_forcing=teacher_forcing, scheduled_sampling_ratio=scheduled_sampling_ratio, train=train)
        # (bsz,34,24,48) (bsz,24)
        if train:
            sst_p_9 = torch.cat([sst_p_all[:, :3, :, :], sst_p_9], dim=1)
        assert sst_p_9.size() == sst_p_all.size()
        sst_p_all = (sst_p_all + sst_p_9) / 2

        nino_p_all = (nino_p + nino_p_half + nino_p_3 + nino_p_6 + nino_p_9) / 5
        return sst_p_all, nino_p_all



if __name__ == "__main__":
    from config import configs
    # configs.input_dim = 3
    # configs.output_dim = 1

    configs.use_hc = 1
    # configs.time = 'no'
    # configs.time = 'stable'
    # configs.time = 'learn'
    configs.time = 'hybrid'

    input_y = torch.rand((2, 38, 1))
    mon = torch.arange(38) % 12
    mon = mon[None, :, None].repeat(2, 1, 1)
    season = torch.arange(38) % 4
    season = season[None, :, None].repeat(2, 1, 1)
    input_y = torch.cat([input_y, mon, season], dim=-1).to(configs.device)

    input_x = torch.rand((2, 38, 24, 48, 4))

    if configs.time == 'no':
        pass  # 默认为2
    elif configs.time == 'stable' or configs.time == 'learn':
        configs.input_dim = 4
    elif configs.time == 'hybrid':
        configs.input_dim = 6

    model = SAConvLSTM(configs).to(configs.device)
    total_num = sum([param.nelement() for param in model.parameters()])
    print('Total params num: {}'.format(total_num))
    print('*****************Finish Parameter****************')
    sst_p, nino_p = model(input_x, input_y, teacher_forcing=True, scheduled_sampling_ratio=0.5, train=True)
    print(sst_p.size())
    print(nino_p.size())

# use_hc = 0
# torch.Size([2, 37, 2, 24, 48])
# torch.Size([2, 24])

# use_hc = 1
# torch.Size([2, 37, 24, 48])
# torch.Size([2, 24])

