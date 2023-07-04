"""
Multi-scale temporal frequency axial attention neural network (MTFAA).

shmzhang@aslp-npu.org, 2022
"""

import torch as th
import torch
import torch.nn as nn
import torch.nn.functional as tf
from typing import List

from .tfcm import TFCM
from .asa import ASA, FASA
from .phase_encoder import PhaseEncoder
from .f_sampling import FD, FU, FDS
from .erb import Banks
from .stft import STFT

def parse_1dstr(sstr: str) -> List[int]:
    return list(map(int, sstr.split(",")))

def parse_2dstr(sstr: str) -> List[List[int]]:
    return [parse_1dstr(tok) for tok in sstr.split(";")]

eps = 1e-10

class MTFAANet(nn.Module):
    def __init__(self,
                 n_sig=1,
                 PEc=4,
                 #Co="48,96,192",
                 #Co="96,96,96",
                 Co=[48,96,92],
                 O="1,1,1",
                 causal=True,
                 bottleneck_layer=2,
                 tfcm_layer=6,
                 mag_f_dim=3,
                 win_len=32*48,
                 win_hop=8*48,
                 nerb=256,
                 sr=16000,
                 win_type="hann",
                 type_encoder = "FD",
                 type_ASA = "ASA"
                 ):
        super(MTFAANet, self).__init__()
        self.PE = PhaseEncoder(PEc, n_sig)
        # 32ms @ 48kHz
        self.stft = STFT(win_len, win_hop, win_len, win_type)
        self.ERB = Banks(nerb, win_len, sr)
        self.encoder_fd = nn.ModuleList()
        self.encoder_bn = nn.ModuleList()
        self.bottleneck = nn.ModuleList()
        self.decoder_fu = nn.ModuleList()
        self.decoder_bn = nn.ModuleList()
        C_en = [PEc//2*n_sig] + Co 
        C_de = [4] + Co
        O = parse_1dstr(O)

        if type_encoder == "FD" : 
            encoder = FD
        elif type_encoder == "FDS" : 
            encoder = FDS
        else :
            raise Exception("Encoder type {} is not defined".format(type_encoder))

        if type_ASA == "ASA"  :
            asa = ASA
        elif type_ASA == "FASA" :
            asa = FASA
        else :
            raise Exception("ASA type {} is not defined".format(type_ASA))

        for idx in range(len(C_en)-1):
            self.encoder_fd.append(
                encoder(C_en[idx], C_en[idx+1]),
            )
            self.encoder_bn.append(
                nn.Sequential(
                    TFCM(C_en[idx+1], (3, 3),
                         tfcm_layer=tfcm_layer, causal=causal),
                    asa(C_en[idx+1], causal=causal),
                )
            )

        for idx in range(bottleneck_layer):
            self.bottleneck.append(
                nn.Sequential(
                    TFCM(C_en[-1], (3, 3),
                         tfcm_layer=tfcm_layer, causal=causal),
                    asa(C_en[-1], causal=causal),
                )
            )

        for idx in range(len(C_de)-1, 0, -1):
            self.decoder_fu.append(
                FU(C_de[idx], C_de[idx-1], O=(O[idx-1], 0)),
            )
            self.decoder_bn.append(
                nn.Sequential(
                    TFCM(C_de[idx-1], (3, 3),
                         tfcm_layer=tfcm_layer, causal=causal),
                    asa(C_de[idx-1], causal=causal),
                )
            )

        # MEA is causal, so mag_t_dim = 1.
        self.mag_mask = nn.Conv2d(
            4, mag_f_dim, kernel_size=(3, 1), padding=(1, 0))
        self.real_mask = nn.Conv2d(4, 1, kernel_size=(3, 1), padding=(1, 0))
        self.imag_mask = nn.Conv2d(4, 1, kernel_size=(3, 1), padding=(1, 0))
        kernel = th.eye(mag_f_dim)
        kernel = kernel.reshape(mag_f_dim, 1, mag_f_dim, 1)
        self.register_buffer('kernel', kernel)
        self.mag_f_dim = mag_f_dim

    def forward(self, X):
        # X : [B,C(2),F,T]
        mag = th.norm(X, dim=1)
        pha = torch.atan2(X[:, -1, ...], X[:, 0, ...])
        out = self.ERB.amp2bank(self.PE([X]))

        encoder_out = []
        for idx in range(len(self.encoder_fd)):
            out = self.encoder_fd[idx](out)
            encoder_out.append(out)
            out = self.encoder_bn[idx](out)

        for idx in range(len(self.bottleneck)):
            out = self.bottleneck[idx](out)

        for idx in range(len(self.decoder_fu)):
            out = self.decoder_fu[idx](out, encoder_out[-1-idx])
            out = self.decoder_bn[idx](out)
        
        out = self.ERB.bank2amp(out)

        # stage 1
        mag_mask = self.mag_mask(out)
        mag_pad = tf.pad(
            mag[:, None], [0, 0, (self.mag_f_dim-1)//2, (self.mag_f_dim-1)//2])
        mag = tf.conv2d(mag_pad, self.kernel)
        mag = mag * mag_mask.sigmoid()
        mag = mag.sum(dim=1)
        # stage 2
        real_mask = self.real_mask(out).squeeze(1)
        imag_mask = self.imag_mask(out).squeeze(1)

        mag_mask = th.sqrt(th.clamp(real_mask**2+imag_mask**2, eps))
        pha_mask = th.atan2(imag_mask+eps, real_mask+eps)
        real = mag * mag_mask.tanh() * th.cos(pha+pha_mask)
        imag = mag * mag_mask.tanh() * th.sin(pha+pha_mask)
        #return mag, th.stack([real, imag], dim=1), self.stft.inverse(real, imag)

        return real + imag*1j
    

class MTFAA_helper(nn.Module):
    def __init__(self,
                 Co=[48,96,192],
                 type_encoder = "FD",
                 type_ASA = "ASA",
                 n_fft = 512,
                 n_hop = 128,
                 n_erb = 64
                 ) :
        super(MTFAA_helper,self).__init__()

        self.n_fft = n_fft
        self.n_hop = n_hop

        self.model = MTFAANet(
            Co=Co,
            win_len= n_fft,
            win_hop = n_hop,
            nerb=n_erb,
            type_encoder=type_encoder,
            type_ASA=type_ASA
        )

    def forward(self, x):
        # STFT
        X = torch.stft(x, n_fft = self.n_fft, window=torch.hann_window(self.n_fft).to(x.device),return_complex=False)

        # X.shape == (B,F,T,2)
        X = torch.permute(X,(0,3,1,2))
        # X.shape == (B,2,F,T)
        Y = self.model(X)

        # iSTFT
        y = torch.istft(Y, self.n_fft, window = torch.hann_window(self.n_fft).to(Y.device))

        return y
    


def test():
    x = torch.rand(2,64000)
    print(x.shape)

    model = MTFAA_helper()

    y = model(x)
    print(y.shape)