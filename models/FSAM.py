# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
# from layers.Transformer_EncDec import Encoder, EncoderLayer
# from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Attention_masks import Mahalanobis_mask, Granger_Causality_mask, GC_prob_mask
from layers.Attention_masks import Encoder, EncoderLayer,FullAttention, AttentionLayer
import numpy as np


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.revin = configs.revin
        self.if_complex = configs.if_complex
        self.output_attention = configs.output_attention

        self.input_total = int(self.seq_len / 2 + 1)
        self.output_total = int((self.seq_len + self.pred_len) / 2 + 1)
        self.total_channels = 2 * self.channels

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector_in = nn.Linear(self.pred_len, configs.d_model, bias=True)
        self.projector_out = nn.Linear(configs.d_model, self.pred_len, bias=True)
        if configs.mask_id == 1:
            self.mask_generator = Mahalanobis_mask(self.seq_len)
        elif configs.mask_id == 2:
            self.mask_generator = Granger_Causality_mask(configs.mask_file, configs.threshold)
        elif configs.mask_id == 3:
            self.mask_generator = GC_prob_mask(configs.mask_file)

        if self.if_complex:
            self.patch_layers = Linear_Complex(configs)
        else:
            self.patch_layers = Linear_Separate(configs)

    def forward(self, x):  # x: [Batch, Input length, Channel]
        if self.revin:
            # RIN
            x_mean = torch.mean(x, dim=1, keepdim=True)
            x = x - x_mean
            x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            # print(x_var)
            x = x / torch.sqrt(x_var)

        x_origin = x.permute(0, 2, 1).clone()
        x_complex = torch.fft.rfft(x, dim=1)

        x_complex = self.patch_layers(x_complex)

        x = torch.fft.irfft(x_complex, dim=1)

        x_out = x[:, -self.pred_len:, :].permute(0, 2, 1)
        enc_in = self.projector_in(x_out)
        
        channel_mask = self.mask_generator(x_origin)
        # print('---------channel_mask:',channel_mask)
        enc_out, attns = self.encoder(enc_in, attn_mask=channel_mask)
        x_cd = self.projector_out(enc_out).permute(0, 2, 1)
        x = x_cd

        # print('----------after irfft:', x.shape)
        if self.revin:
            ans_x = (x) * torch.sqrt(x_var) + x_mean
        else:
            ans_x = x
            x_var = x
        return ans_x, (x) * torch.sqrt(x_var)


class Linear_Complex(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = configs.individual
        self.channels = configs.enc_in
        self.affine = configs.affine
        self.main_freq = configs.main_freq
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_total = int(self.seq_len / 2 + 1)
        self.output_total = int((self.seq_len + self.pred_len) / 2 + 1)
        self.patch_num = (self.output_total - self.patch_len) // self.stride + 1

        self.patch_emb = PatchEmbed1D(self.patch_len, self.stride)
        self.patch_reconstruct = PatchReconstruct1D(self.patch_len, self.stride)

        self.freq_upsampler = nn.Linear(self.input_total, self.output_total).to(
            torch.cfloat)  # complex layer for frequency upcampling]
        
        self.gelu = ComplexGELU()

        real_w = torch.empty(self.patch_num, self.patch_len, self.patch_len)
        imag_w = torch.empty(self.patch_num, self.patch_len, self.patch_len)
        nn.init.xavier_uniform_(real_w)
        nn.init.xavier_uniform_(imag_w)
        self.patch_weight = nn.Parameter(torch.complex(real_w, imag_w)).to(self.device)
        # [num_patches, patch_len, patch_len]

        real_b = torch.zeros(self.patch_num, 1, self.patch_len)
        imag_b = torch.zeros(self.patch_num, 1, self.patch_len)
        self.patch_bias = nn.Parameter(torch.complex(real_b, imag_b)).to(self.device)
        # [num_patches, 1, patch_len] for broadcasting

    def forward(self, x):  # x: [Batch, input_total, Channel]

        total_upsample_ = torch.zeros(
            [x.size(0), self.output_total, x.size(2)],
            dtype=x.dtype).to(x.device)
        patch_out = torch.zeros(  # patch_out: [Batch,Channel,patch_num,patch_len]
            [x.size(0), self.channels, self.patch_num, self.patch_len],
            dtype=x.dtype).to(x.device)

        # [Batch,Channel,output_total]
        total_upsample_ = self.freq_upsampler(x.permute(0, 2, 1))
        # total_upsample_ = total_upsample_.permute(0, 2, 1)  # [Batch,Channel,output_total]
        # input_batch: [Batch,Channel,patch_num,patch_len]
        input_batch = self.patch_emb(total_upsample_)
        # input_batch = total_upsample_.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # print('----------before patch:', total_upsample_.shape)
        # print('----------after patch:', input_batch.shape)

        B, C, P, D = input_batch.shape
        input_batch = input_batch.reshape(B * C, P, D)
        input_batch = input_batch.permute(1, 0, 2)  # -> [patch_num, b*c, patch_len]

        # patch_out -> [patch_num, b*c, patch_len]
        patch_out = torch.einsum("pbd,pdk->pbk", input_batch, self.patch_weight) + self.patch_bias
        # patch_out = torch.matmul(input_batch, self.patch_weight) + self.patch_bias
        patch_out = patch_out.permute(1, 0, 2).contiguous().reshape(B, C, P, D)

        # for j in range(self.patch_num):
        #     patch_out[:, :, j, :] = self.patch_layers[j](input_batch[:, :, j, :])

        #  fold
        batch_size = patch_out.shape[0]
        c_in = patch_out.shape[1]
        patch_num = patch_out.shape[2]
        patch_len = patch_out.shape[3]

        output_size = self.output_total

        # output_folded = self.patch_reconstruct(patch_out, output_size)      
        patch_out_r = patch_out.real
        patch_out_i = patch_out.imag

        zr_folded = F.fold(
            patch_out_r.permute(0, 1, 3, 2).reshape(batch_size, -1, patch_num),
            output_size=(1, output_size),
            kernel_size=(1, patch_len),
            stride=(1, self.stride)
        )
        zi_folded = F.fold(
            patch_out_i.permute(0, 1, 3, 2).reshape(batch_size, -1, patch_num),
            output_size=(1, output_size),
            kernel_size=(1, patch_len),
            stride=(1, self.stride)
        )
        #
        overlap_counts = F.fold(
            torch.ones_like(patch_out).permute(0, 1, 3, 2).reshape(batch_size, -1, patch_num),
            output_size=(1, output_size),
            kernel_size=(1, patch_len),
            stride=(1, self.stride)
        )
        # output_folded [Batch,Channel,output_total]
        z1_folded = torch.complex(zr_folded, zi_folded)
        overlap_counts[overlap_counts == 0] = 1
        output_folded = z1_folded / overlap_counts  #
        output_folded = output_folded.view(batch_size, self.channels, output_size)  #
        output_folded = output_folded.permute(0, 2, 1)
        return output_folded


class Linear_Separate(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = configs.individual
        self.channels = configs.enc_in
        self.affine = configs.affine
        self.main_freq = configs.main_freq
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_total = int(self.seq_len / 2 + 1)
        self.output_total = int((self.seq_len + self.pred_len) / 2 + 1)
        self.patch_num = (self.output_total - self.patch_len) // self.stride + 1

        self.patch_emb = PatchEmbed1D(self.patch_len, self.stride)
        self.patch_reconstruct = PatchReconstruct1D(self.patch_len, self.stride)

        self.freq_upsampler_real = nn.Linear(self.input_total, self.output_total)
        self.freq_upsampler_imag = nn.Linear(self.input_total, self.output_total)        
        self.gelu = ComplexGELU()

        self.weight_real_init = torch.empty(self.patch_num, self.patch_len, self.patch_len)
        self.weight_imag_init = torch.empty(self.patch_num, self.patch_len, self.patch_len)
        nn.init.xavier_uniform_(self.weight_real_init)
        nn.init.xavier_uniform_(self.weight_imag_init)
        # [num_patches, patch_len, patch_len]

        self.bias_real_init = torch.zeros(self.patch_num, 1, self.patch_len)
        self.bias_imag_init = torch.zeros(self.patch_num, 1, self.patch_len)
        # [num_patches, 1, patch_len] for broadcasting
        
        self.weight_real = nn.Parameter(self.weight_real_init)
        # [num_patches, patch_len, patch_len]
        self.bias_real = nn.Parameter(self.bias_real_init)
        # [num_patches, 1, patch_len] for broadcasting
        self.weight_imag = nn.Parameter(self.weight_imag_init)
        # [num_patches, patch_len, patch_len]
        self.bias_imag = nn.Parameter(self.bias_imag_init)
        # [num_patches, 1, patch_len] for broadcasting

    def forward(self, x):  # x: [Batch, input_total, Channel]
        x_r = x.real
        x_i = x.imag

        total_upsample_real = torch.zeros(
            [x_r.size(0), self.output_total, x_r.size(2)],
            dtype=x_r.dtype).to(x_r.device)
        total_upsample_imag = torch.zeros(
            [x_i.size(0), self.output_total, x_i.size(2)],
            dtype=x_i.dtype).to(x_i.device)
        patch_out_real = torch.zeros(  # patch_out: [Batch,Channel,patch_num,patch_len]
            [x_r.size(0), self.channels, self.patch_num, self.patch_len],
            dtype=x_r.dtype).to(x_r.device)
        patch_out_imag = torch.zeros(  # patch_out: [Batch,Channel,patch_num,patch_len]
            [x_i.size(0), self.channels, self.patch_num, self.patch_len],
            dtype=x_i.dtype).to(x_i.device)

        # [Batch,Channel,output_total]
        total_upsample_real = self.freq_upsampler_real(x_r.permute(0, 2, 1))
        total_upsample_imag = self.freq_upsampler_imag(x_i.permute(0, 2, 1))

        # total_upsample_ = total_upsample_.permute(0, 2, 1)  # [Batch,Channel,output_total]
        # input_batch: [Batch,Channel,patch_num,patch_len]
        input_batch_real = self.patch_emb(total_upsample_real)
        input_batch_imag = self.patch_emb(total_upsample_imag)

        # input_batch_real = total_upsample_real.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # input_batch_imag = total_upsample_imag.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # print('----------before patch:', total_upsample_real.shape)
        # print('----------after patch:', input_batch_real.shape)
        B, C, P, D = input_batch_real.shape
        input_batch_real = input_batch_real.reshape(B * C, P, D)
        input_batch_imag = input_batch_imag.reshape(B * C, P, D)
        input_batch_real = input_batch_real.permute(1, 0, 2)
        input_batch_imag = input_batch_imag.permute(1, 0, 2)  # -> [patch_num, b*c, patch_len]
        # print('-----input_batch_real-----',input_batch_real.shape)
        # print('-----weight_real-----',self.weight_real.shape)
        # print('-----bias_real-----',self.bias_real.shape)

        # patch_out -> [patch_num, b*c, patch_len]
        patch_out_real = torch.einsum("pbd,pdk->pbk", input_batch_real, self.weight_real) + self.bias_real
        patch_out_imag = torch.einsum("pbd,pdk->pbk", input_batch_imag, self.weight_imag) + self.bias_imag
        # patch_out_real = torch.matmul(input_batch_real, self.weight_real) + self.bias_real
        # patch_out_imag = torch.matmul(input_batch_imag, self.weight_imag) + self.bias_imag
        patch_out_real = patch_out_real.permute(1, 0, 2).contiguous().reshape(B, C, P, D)
        patch_out_imag = patch_out_imag.permute(1, 0, 2).contiguous().reshape(B, C, P, D)

        # for j in range(self.patch_num):
        #     patch_out_real[:, :, j, :] = self.patch_layers_real[j](input_batch_real[:, :, j, :])
        #     patch_out_imag[:, :, j, :] = self.patch_layers_imag[j](input_batch_imag[:, :, j, :])

        input_batch = torch.complex(input_batch_real, input_batch_imag)
        patch_out = torch.complex(patch_out_real, patch_out_imag)
        #  fold
        batch_size = patch_out.shape[0]
        c_in = patch_out.shape[1]
        patch_num = patch_out.shape[2]
        patch_len = patch_out.shape[3]

        output_size = self.output_total

        # output_folded = self.patch_reconstruct(patch_out, output_size)
        
        zr_folded = F.fold(
            patch_out_real.permute(0, 1, 3, 2).reshape(batch_size, -1, patch_num),
            output_size=(1, output_size),
            kernel_size=(1, patch_len),
            stride=(1, self.stride)
        )
        zi_folded = F.fold(
            patch_out_imag.permute(0, 1, 3, 2).reshape(batch_size, -1, patch_num),
            output_size=(1, output_size),
            kernel_size=(1, patch_len),
            stride=(1, self.stride)
        )
        #
        overlap_counts = F.fold(
            torch.ones_like(patch_out).permute(0, 1, 3, 2).reshape(batch_size, -1, patch_num),
            output_size=(1, output_size),
            kernel_size=(1, patch_len),
            stride=(1, self.stride)
        )
        # output_folded [Batch,Channel,output_total]
        z1_folded = torch.complex(zr_folded, zi_folded)
        overlap_counts[overlap_counts == 0] = 1
        output_folded = z1_folded / overlap_counts  #
        output_folded = output_folded.view(batch_size, self.channels, output_size)  #
        output_folded = output_folded.permute(0, 2, 1)
        return output_folded


class PatchEmbed1D(nn.Module):
    def __init__(self, patch_len, stride):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, x):
        # x: [B, C, L]
        B, C, L = x.shape
        # total_len = ((L - self.patch_len + self.stride - 1) // self.stride) * self.stride + self.patch_len
        total_len = L + self.patch_len - 1

        pad_len = total_len - L

        # x_padded = F.pad(x, (0, pad_len))  # pad on the right
        patches = x.unfold(2, self.patch_len, self.stride)  # [B, C, N, patch_len]

        return patches  #


class PatchReconstruct1D(nn.Module):
    def __init__(self, patch_len, stride):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, patches, original_len):
        # patches: [B, C, N, patch_len_new]
        B, C, N, K = patches.shape
        # total_len = ((original_len - self.patch_len + self.stride - 1) // self.stride) * self.stride + self.patch_len
        total_len = N * self.stride + K

        result = torch.zeros(
            [B, C, total_len],
            dtype=patches.dtype).to(patches.device)
        count = torch.zeros(
            [B, C, total_len],
            dtype=patches.dtype).to(patches.device)
        # result = torch.zeros(B, C, total_len, device=patches.device)
        # count = torch.zeros(B, C, total_len, device=patches.device)

        for i in range(N):
            start = i * self.stride
            end = start + K
            if end > total_len:
                overlap_len = total_len - start
                result[:, :, start:] += patches[:, :, i, :overlap_len]
                count[:, :, start:] += 1
            else:
                result[:, :, start:end] += patches[:, :, i, :]
                count[:, :, start:end] += 1

        count[count == 0] = 1
        reconstructed = result / count
        return reconstructed[:, :, :original_len]  #
    
class ComplexGELU(nn.Module):
    def __init__(self):
        super(ComplexGELU, self).__init__()

    def forward(self, x):
        if not torch.is_complex(x):
            raise ValueError("Input must be a complex tensor")
        real = F.gelu(x.real)
        imag = F.gelu(x.imag)
        return torch.complex(real, imag)
    