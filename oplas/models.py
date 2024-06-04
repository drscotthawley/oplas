import torch 
import torch.nn as nn
import os 
import laion_clap
import numpy as np 
import torchaudio
from torchaudio.prototype.pipelines import VGGISH

#------ first, our Projector model that is made of a bunch of EmbedBlocks -------

class EmbedBlock(nn.Module):
    def __init__(self, in_dims:int, out_dims:int, act=nn.GELU(), resid=True, use_bn=False, requires_grad=True, **kwargs) -> None:
        "generic little block for embedding stuff.  note residual-or-not doesn't seem to make a huge difference for a-a"
        super().__init__()
        self.in_dims, self.out_dims, self.act, self.resid = in_dims, out_dims, act, resid
        self.lin = nn.Linear(in_dims, out_dims, **kwargs)
        self.bn = nn.BatchNorm1d(out_dims) if use_bn else None # even though points in 2d, only one non-batch dim in data
        self.dropout = nn.Dropout(0.001)  

        if requires_grad == False:
            self.lin.weight.requires_grad = False
            self.lin.bias.requires_grad = False

    def forward(self, xin):
        x = self.lin(xin)
        x = self.dropout(x) if self.dropout is not None else x
        if self.bn  is not None: x = self.bn(x)  # vicreg paper uses BN before activation
        if self.act is not None: x = self.act(x)
        #if self.bn  is not None: x = self.bn(x)   # re. "BN before or after Activation? cf. https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md"
        return xin + x if (self.resid and self.in_dims==self.out_dims) else x 


class Projector(nn.Module):
    """
    Main Projector model. Patterned after VICReg's simple MLP
    """
    def __init__(self, 
                 in_dims=128, 
                 out_dims=128,
                 hidden_dims_scale=4, 
                 num_inner_layers=6,
                 act=nn.GELU(), 
                 use_bn=False,  # bn is bad for regression model
                 resid=True, 
                 block=EmbedBlock, # Linear layer with optional activation & optional BatchNorm
                 trivial=False,   # ignore everything and make this an identity mapping
                ):
        super().__init__()
        self.resid, self.trivial = resid, trivial
        hidden_dims = hidden_dims_scale * in_dims
        #resid=False # turn it off for inner layers, just leave outer resid
        self.encoder = nn.Sequential(  
            block( in_dims,     hidden_dims, act=act,  use_bn=use_bn, resid=resid),
            *[block(hidden_dims, hidden_dims, act=act, use_bn=use_bn, resid=resid) for _ in range(num_inner_layers)],
            block( hidden_dims, out_dims,    act=None, use_bn=use_bn, resid=resid, bias=False), # bias=False from VICReg paper
        )
        self.decoder = nn.Sequential(  # same as encoder, in fact. 
            block( out_dims,    hidden_dims, act=act,  use_bn=use_bn, resid=resid),
            *[block(hidden_dims, hidden_dims, act=act, use_bn=use_bn, resid=resid) for _ in range(num_inner_layers)],   
            block( hidden_dims, in_dims,     act=None, use_bn=use_bn, resid=resid),
        )
            
    def encode(self,y):
        if self.trivial: return y  # "trivial" no-op  flag for quick testing
        z = self.encoder(y) # transpose is just so embeddings dim goes last for matrix mult
        return z + y if self.resid else z

    def decode(self,z):
        if self.trivial: return z 
        y = self.decoder(z)# .transpose(1,2)).transpose(1,2)
        return y + z if self.resid else y

    def forward(self, 
        y   # the 'encodings' vector from the given encoder
        ):
        z = self.encode(y)
        y_hat = self.decode(z)  # train projector to approximately invert itself (and hope it doesn't all collapse to nothing!)
        return z, y_hat  # encoder output,  decoder output


# -- Now a list of "given models" i.e. pretrained encoders (CLAP, Vggish, etc)


class CLAPEncoder(nn.Module):
    def __init__(self, enable_fusion=True, **kwargs):
        super().__init__()
        self.model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, **kwargs)
        self.model.load_ckpt()
        self.model.eval()
        self.sample_rate = 48000

    @torch.no_grad()
    def encode(self, audio):
        while len(audio.shape) < 3: 
            audio = audio.unsqueeze(0) # add batch and/or channel dims 
        if audio.shape[-1]==2:         # stereo to mono: average TODO: make this more robust
            audio = audio.mean(dim=-1)
        encodings = self.model.get_audio_embedding_from_data(x=audio, use_tensor=True).to(audio.dtype)
        return encodings

    def forward(self, audio):
        return self.encode(audio)


# class VGGishEncoder_Old(nn.Module):
#     """
#     THIS VERSION needs NumpyArrays and has been replaced with the one below
#     We use the .forward of the VGGISH torchaudio model, not the one from harritaylor on torch.hub
#     input (torch.Tensor) – batch of spectrograms, with shape (n_example, 1, n_frame, 64).
#     """
#     def __init__(self, accelerator=None, **kwargs):
#         super().__init__()
#         encoder = torch.hub.load('harritaylor/torchvggish', 'vggish')
#         #encoder = torchaudio.prototype.pipelines.VGGISH
#         use_pca = False
#         use_activation = False
#         if not use_pca:  encoder.postprocess = False
#         if not use_activation: encoder.embeddings = torch.nn.Sequential(*list(encoder.embeddings.children())[:-1])
#         self.sample_rate = 16000
#         self.encoder = encoder
#         self.encoder.eval()
#         self.accelerator = accelerator


#     def encode(self, audio, time_avg=True, debug=False):
#         # NOTE:  the numpy nature of this makes it slow AF but VGGish is numpy-only :shrug: 
#         if debug: print("audio.shape =",audio.shape)
#         audio = torch.mean(audio, dim=-1)   # vggish requries we convert to mono
#         encodings = []                    # ...whoa, vggish can't even handle batches?  we have to pass 'em through singly?
#         for bi, waveform in enumerate(audio):  # TODO speed this up!!
#             if debug: print("waveform.shape =",waveform.shape)
#             # yikes 'torchvggish' [sic] requires *numpy* (not torch) inputs :thumbs-down: 
#             if self.accelerator is not None: 
#                 e =  self.accelerator.unwrap_model(self.encoder).forward(waveform.cpu().numpy(), self.sample_rate)
#             else: 
#                 e = self.encoder.forward(waveform.cpu().numpy(), self.sample_rate) 
#             encodings.append(e) 
#         encodings = torch.cat(encodings, dim=0)
#         if time_avg: 
#             if debug: print("pre-time_avg: encodings.shape =",encodings.shape)
#             embeddings = torch.mean(encodings, dim=-2)
#             if debug: print("pre-time_avg: encodings.shape =",encodings.shape)
#         return encodings.to(audio.device)

#     def forward(self, audio):
#         return self.encode(audio)



class VGGishEncoder(nn.Module):
    """
    cf. https://pytorch.org/audio/main/generated/torchaudio.prototype.pipelines.VGGishBundle.html
    """
    def __init__(self):
        super().__init__()
        self.sample_rate = VGGISH.sample_rate
        self.input_proc = VGGISH.get_input_processor()
        self.encoder = VGGISH.get_model()
        self.encoder.eval()

    @torch.no_grad()
    def encode(self, audio, time_avg=True, debug=False):
        if debug: print("\nVGGishEncoder.encode: audio.shape =",audio.shape)
        audio = torch.mean(audio, dim=-1)   # vggish requries we convert to mono
        #encodings = self.encoder(self.input_proc(audio))  # fully batched version. WON'T WORK
        # VGGish pipeline can't handle batches so we need to send them one at a time....
        encodings = torch.empty(audio.shape[0], 128, device=audio.device, dtype=audio.dtype)
        for bi, waveform in enumerate(audio): 
            if debug: print("   waveform.shape =",waveform.shape) 
            inp = self.input_proc(waveform)
            if debug: print("   inp.shape =",inp.shape)
            e = self.encoder(inp) #, self.sample_rate) 
            if debug: print("   e.shape =",e.shape)
            if time_avg: e = torch.mean(e, dim=-2) 
            encodings[bi] = e
        if debug: print("   encodings.shape =",encodings.shape)
        return encodings 

    def forward(self, audio):
        return self.encode(audio)