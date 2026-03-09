#-*- coding:utf-8 -*-
#
# Original code is here: https://github.com/openai/guided-diffusion
#
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .modules import *
import matplotlib.pyplot as plt

NUM_CLASSES = 1
def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    # from VQGAN
    return self

class model_ensemble(nn.Module):
    def __init__(self, model_3D, model2D_2, model2D_3, 
                 batch_size_2D_inference = 8, 
                 time_step = 1000, 
                 out_channels = 1, 
                 ntime_steps_2D = 1000, 
                 baseline = "3D"
                 ):
        super(model_ensemble, self).__init__()
        self.ntime_steps_2D = ntime_steps_2D
        self.model_3D = model_3D
        self.model2D_2 = model2D_2.netG
        self.model2D_3 = model2D_3.netG
        self.baseline = baseline
        # TODO change this from hard coding
        self.featue_c = 64
        # if model2D_3 is a DDP model, we need to access the module
        if hasattr(self.model2D_3, "module"):
            self.model2D_3 = self.model2D_3.module
            self.model2D_2 = self.model2D_2.module
            
        self.batch_size_2D_inference = batch_size_2D_inference
        self.time_step = time_step
        if (192 % batch_size_2D_inference != 0) or (152 % batch_size_2D_inference != 0):
            print("invalid batch size 2D",batch_size_2D_inference)
            raise ValueError("batch_size_2D_inference must be a factor of 192 and 152")
        self.out_c = out_channels
        self.model2D_2.denoise_fn.eval()
        self.model2D_3.denoise_fn.eval()
        for param in self.model2D_2.denoise_fn.parameters():
            param.requires_grad = False
        for param in self.model2D_3.denoise_fn.parameters():
            param.requires_grad = False
        
        if baseline == "3D_feature":
            # self.feature_layers = []
            # zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1, padding_mode = 'replicate'))
            # self.feature_layer_1_1 = zero_module(nn.Conv3d(512, 192, (8,1,1), stride = (8,1,1)))
            # self.feature_layer_1_4 = zero_module(nn.Conv3d(256, 128, (4,1,1), stride = (4,1,1)))
            # self.feature_layer_1_7 = zero_module(nn.Conv3d(128, 64, (2,1,1), stride = (2,1,1)))
            
            self.feature_layer_2_1 = zero_module(nn.Conv3d(512, 192, (1,8,1), stride = (1,8,1)))
            self.feature_layer_2_4 = zero_module(nn.Conv3d(256, 192, (1,4,1), stride = (1,4,1)))
            self.feature_layer_2_7 = zero_module(nn.Conv3d(128, 128, (1,2,1), stride = (1,2,1)))
            
            self.feature_layer_3_1 = zero_module(nn.Conv3d(512, 192, (1,1,8), stride = (1,1,8)))
            self.feature_layer_3_4 = zero_module(nn.Conv3d(256, 192, (1,1,4), stride = (1,1,4)))
            self.feature_layer_3_7 = zero_module(nn.Conv3d(128, 128, (1,1,2), stride = (1,1,2)))
            
            # learning layers, this makes it possible to just sum them up!
            # self.feature_extract_layer_1_1 = zero_module(nn.Conv3d(192, 192, 3, stride = 1, padding = 1))
            # self.feature_extract_layer_1_4 = zero_module(nn.Conv3d(128, 128, 3, stride = 1, padding = 1))
            # self.feature_extract_layer_1_7 = zero_module(nn.Conv3d(128, 128, 3, stride = 1, padding = 1))
            # self.feature_extract_layer_1_10 = zero_module(nn.Conv3d(64, 64, 3, stride = 1, padding = 1))
            
            # self.feature_extract_layer_2_1 = zero_module(nn.Conv3d(256, 192, 3, stride = 1, padding = 1))
            # self.feature_extract_layer_2_4 = zero_module(nn.Conv3d(256, 256, 3, stride = 1, padding = 1))
            # self.feature_extract_layer_2_7 = zero_module(nn.Conv3d(128, 128, 3, stride = 1, padding = 1))
            self.feature_extract_layer_2_10 = zero_module(nn.Conv3d(64, 64, 1, stride = 1, padding = 0))
            
            # self.feature_extract_layer_3_1 = zero_module(nn.Conv3d(256, 192, 3, stride = 1, padding = 1))
            # self.feature_extract_layer_3_4 = zero_module(nn.Conv3d(256, 256, 3, stride = 1, padding = 1))
            # self.feature_extract_layer_3_7 = zero_module(nn.Conv3d(128, 128, 3, stride = 1, padding = 1))
            self.feature_extract_layer_3_10 = zero_module(nn.Conv3d(64, 64, 1, stride = 1, padding = 0))

        else:
            self.feature_layers = None
        # This only works for pl.lightning.LightningModule
        # self.model2D_2.denoise_fn = self.model2D_2.denoise_fn.eval()
        # self.model2D_3.denoise_fn.train = disabled_train
        # self.model2D_3.denoise_fn = self.model2D_3.denoise_fn.eval()
        # self.model2D_3.denoise_fn.train = disabled_train
    def debug_viz(self, arr, name = "debug/viz"):
        # import matplotlib.pyplot as plt
        # nor arr to 0-1
        print("saving img in ", name)
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = arr.cpu().numpy()
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                plt.imshow(arr[i,j,:,:])
                plt.savefig(f"{name}_{i}_{j}.png")

    def debug_viz_3D(self, arr, name = "debug/viz"):
        # import matplotlib.pyplot as plt
        # nor arr to 0-1
        print("saving img in ", name)
        # if len(shape) is not 5, we need to add dimension until it's 5:
        if len(arr.shape) == 3:
            arr = arr.unsqueeze(1)
        if len(arr.shape) == 4:
            arr = arr.unsqueeze(1)

        center = arr.shape[2] // 2
        # arr = arr.cpu().numpy()
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                cur_arr = arr[i,j,center,:,:]
                cur_arr = (cur_arr - cur_arr.min()) / (cur_arr.max() - cur_arr.min())
                cur_arr = cur_arr.cpu().numpy()
                plt.imshow(cur_arr)
                plt.savefig(f"{name}_{i}_{j}.png")
                
    def forward(self, x, timesteps):
        # x shape is (batch_size, c, 152, 192, 192)
        # print("debugging t for 3D:", timesteps)
        b, c, d, h, w = x.shape
        b_2D = self.batch_size_2D_inference
        # print(timesteps)
        if self.baseline == "dummy":
            return x[:,0:1]
        with torch.no_grad():
            if self.baseline == "3D_only":
                # do not need to do 2D inference if we only use 3D model
                pass
            elif not self.baseline == "3D_feature":
                # no feature is calculated here
                x_2D_2 = x.permute(0, 3, 1, 2, 4).reshape(-1, c, d, w)
                t_2D = torch.full((b*h,), int(timesteps.item()/self.time_step*self.ntime_steps_2D), device=x_2D_2.device, dtype=torch.long)

                out_2D_2 = torch.zeros(b*h, self.out_c, d, w, device=x_2D_2.device, dtype=x_2D_2.dtype)
                
                for i in range(0, b*h, b_2D):
                    out_2D_2[i:i+b_2D] = self.model2D_2.denoise_inference(x_2D_2[i:i+b_2D,:self.out_c], t_2D[i:i+b_2D], y_cond=x_2D_2[i:i+b_2D,self.out_c:]) # 8 1 88 64
                    # save the output of 2D model
                    # self.debug_viz( out_2D_2[i:i+b_2D], f"debug/output_2D_2_{i}")
                out_2D_2 = out_2D_2.reshape(b, h, self.out_c, d, w).permute(0, 2, 3, 1, 4)
                
                x_2D_3 = x.permute(0, 4, 1, 2, 3).reshape(-1, c, d, h)
                out_2D_3 = torch.zeros(b*w, self.out_c, d, h, device=x_2D_3.device, dtype=x_2D_3.dtype)
                for i in range(0, b*h, b_2D):
                    out_2D_3[i:i+b_2D] = self.model2D_3.denoise_inference(x_2D_3[i:i+b_2D,:self.out_c], t_2D[i:i+b_2D], y_cond=x_2D_3[i:i+b_2D,self.out_c:]) # 8 1 88 64
                out_2D_3 = out_2D_3.reshape(b, w, self.out_c, d, h).permute(0, 2, 3, 4, 1)
            else: 
                # 2D_features are calculated here
                x_2D_2 = x.permute(0, 3, 1, 2, 4).reshape(-1, c, d, w)
                t_2D = torch.full((b*h,), int(timesteps.item()/self.time_step*self.ntime_steps_2D), device=x_2D_2.device, dtype=torch.long)

                out_2D_2 = torch.zeros(b*h, self.out_c, d, w, device=x_2D_2.device, dtype=x_2D_2.dtype)
                # feature_2D_2 = torch.zeros(b*h, self.featue_c, d, w, device=x_2D_2.device, dtype=x_2D_2.dtype)
                out_feature_2_1 = torch.zeros(b*h, 512, 19, 24, device=x_2D_2.device, dtype=x_2D_2.dtype)
                out_feature_2_4 = torch.zeros(b*h, 256, 38, 48, device=x_2D_2.device, dtype=x_2D_2.dtype)
                out_feature_2_7 = torch.zeros(b*h, 128, 76, 96, device=x_2D_2.device, dtype=x_2D_2.dtype)
                out_feature_2_10 = torch.zeros(b*h, 64, 152, 192, device=x_2D_2.device, dtype=x_2D_2.dtype)
                
                for i in range(0, b*h, b_2D):
                    out_2D_2[i:i+b_2D], F = self.model2D_2.denoise_inference(x_2D_2[i:i+b_2D,:self.out_c], t_2D[i:i+b_2D], y_cond=x_2D_2[i:i+b_2D,self.out_c:], feature = True) # 8 1 88 64
                    # feature_2D_2[i:i+b_2D] = F[-2]
                    out_feature_2_1[i:i+b_2D] = F[1]
                    out_feature_2_4[i:i+b_2D] = F[4]
                    out_feature_2_7[i:i+b_2D] = F[7]
                    out_feature_2_10[i:i+b_2D] = F[10]   

                out_2D_2 = out_2D_2.reshape(b, h, self.out_c, d, w).permute(0, 2, 3, 1, 4)
                # feature_2D_2 = feature_2D_2.reshape(b, h, self.featue_c, d, w).permute(0, 2, 3, 1, 4)
                out_feature_2_1 = out_feature_2_1.reshape(b, h, 512, 19, 24).permute(0, 2, 3, 1, 4)
                out_feature_2_4 = out_feature_2_4.reshape(b, h, 256, 38, 48).permute(0, 2, 3, 1, 4)
                out_feature_2_7 = out_feature_2_7.reshape(b, h, 128, 76, 96).permute(0, 2, 3, 1, 4)
                out_feature_2_10 = out_feature_2_10.reshape(b, h, 64, 152, 192).permute(0, 2, 3, 1, 4)
                
                
                x_2D_3 = x.permute(0, 4, 1, 2, 3).reshape(-1, c, d, h)
                out_2D_3 = torch.zeros(b*w, self.out_c, d, h, device=x_2D_3.device, dtype=x_2D_3.dtype)
                out_feature_3_1 = torch.zeros(b*w, 512, 19, 24, device=x_2D_2.device, dtype=x_2D_2.dtype)
                out_feature_3_4 = torch.zeros(b*w, 256, 38, 48, device=x_2D_2.device, dtype=x_2D_2.dtype)
                out_feature_3_7 = torch.zeros(b*w, 128, 76, 96, device=x_2D_2.device, dtype=x_2D_2.dtype)
                out_feature_3_10 = torch.zeros(b*w, 64, 152, 192, device=x_2D_2.device, dtype=x_2D_2.dtype)
                # feature_2D_3 = torch.zeros(b*w, self.featue_c, d, h, device=x_2D_3.device, dtype=x_2D_3.dtype)
                for i in range(0, b*h, b_2D):
                    out_2D_3[i:i+b_2D], F = self.model2D_3.denoise_inference(x_2D_3[i:i+b_2D,:self.out_c], t_2D[i:i+b_2D], y_cond=x_2D_3[i:i+b_2D,self.out_c:], feature = True) # 8 1 88 64
                    # feature_2D_3[i:i+b_2D] = F[-1]
                    out_feature_3_1[i:i+b_2D] = F[1]
                    out_feature_3_4[i:i+b_2D] = F[4]
                    out_feature_3_7[i:i+b_2D] = F[7]
                    out_feature_3_10[i:i+b_2D] = F[10]

                out_2D_3 = out_2D_3.reshape(b, w, self.out_c, d, h).permute(0, 2, 3, 4, 1)
                # feature_2D_3 = feature_2D_3.reshape(b, w, self.featue_c, d, h).permute(0, 2, 3, 4, 1)
                out_feature_3_1 = out_feature_3_1.reshape(b, w, 512, 19, 24).permute(0, 2, 3, 4, 1)
                out_feature_3_4 = out_feature_3_4.reshape(b, w, 256, 38, 48).permute(0, 2, 3, 4, 1)
                out_feature_3_7 = out_feature_3_7.reshape(b, w, 128, 76, 96).permute(0, 2, 3, 4, 1)
                out_feature_3_10 = out_feature_3_10.reshape(b, w, 64, 152, 192).permute(0, 2, 3, 4, 1)
                
        if self.baseline == "2D":
            w2 = 0.0
            w3 = 1.0
            return w2 * out_2D_2 + w3 * out_2D_3
        elif self.baseline == "merge":
            w2 = 0.5
            w3 = 0.5
            return w2 * out_2D_2 + w3 * out_2D_3
        elif self.baseline == "TPDM":
            w2 = 0.5
            # randeomly pick one of the 2D model to use
            if np.random.rand() > w2:
                return out_2D_2
            else:
                return out_2D_3 
        elif self.baseline == "3D":
            # inference for 3D model
            x_3D = torch.cat([x, out_2D_2.detach(), out_2D_3.detach()], dim = 1)
            ensemble_w = self.model_3D(x_3D, timesteps)
            out_3D = ensemble_w * out_2D_2 + (1 - ensemble_w) * out_2D_3
            return out_3D
        elif self.baseline == "3D_only":
            # inference for 3D model only
            out_3D = self.model_3D(x, timesteps)
            return out_3D
        elif self.baseline == "3D_feature":
            # inference for 3D model
            
                
            out_feature_2_1 = self.feature_layer_2_1(out_feature_2_1)
            out_feature_2_4 = self.feature_layer_2_4(out_feature_2_4)
            out_feature_2_7 = self.feature_layer_2_7(out_feature_2_7)
            
            out_feature_3_1 = self.feature_layer_3_1(out_feature_3_1)
            out_feature_3_4 = self.feature_layer_3_4(out_feature_3_4)
            out_feature_3_7 = self.feature_layer_3_7(out_feature_3_7)
            
            out_feature_1 = out_feature_2_1 + out_feature_3_1
            out_feature_4 = out_feature_2_4 + out_feature_3_4
            out_feature_7 = out_feature_2_7 + out_feature_3_7
            out_feature_10 = self.feature_extract_layer_2_10(out_feature_2_10) + self.feature_extract_layer_3_10(out_feature_3_10)
            out_features = [out_feature_1,out_feature_4,out_feature_7,out_feature_10]
            
            
            x_3D = torch.cat([x, out_2D_2.detach(), out_2D_3.detach()], dim = 1)
            ensemble_w = self.model_3D(x_3D, timesteps, feature_2D = out_features)
            # ensemble_w = self.model_3D(x_3D, timesteps, feature_2D = feature_3D)
            out_3D = ensemble_w * out_2D_2 + (1 - ensemble_w) * out_2D_3
            return out_3D
        else:
            raise ValueError("baseline must be 2D, 3D or merge")


class model_ensemble_2cond(nn.Module):
    def __init__(self, model_3D, model2D_2, model2D_3,
                 model_cond2_2, model_cond2_3, 
                 batch_size_2D_inference = 8, 
                 time_step = 1000, 
                 out_channels = 1, 
                 ntime_steps_2D = 1000, 
                 baseline = "3D_feature"):
        super(model_ensemble_2cond, self).__init__()
        self.ntime_steps_2D = ntime_steps_2D
        self.model_3D = model_3D
        self.model2D_2 = model2D_2.netG
        self.model2D_3 = model2D_3.netG
        self.model_cond2_2 = model_cond2_2.netG
        self.model_cond2_3 = model_cond2_3.netG
        self.sqrt_alphas_cumprod = self.register_buffer("sqrt_alphas_cumprod", torch.zeros(1000))
        
        self.baseline = baseline
        # TODO change this from hard coding
        self.featue_c = 64
        # if model2D_3 is a DDP model, we need to access the module
        if hasattr(self.model2D_3, "module"):
            self.model2D_3 = self.model2D_3.module
            self.model2D_2 = self.model2D_2.module
            self.model_cond2_2 = self.model_cond2_2.module
            self.model_cond2_3 = self.model_cond2_3.module
            
        self.batch_size_2D_inference = batch_size_2D_inference
        self.time_step = time_step
        if (192 % batch_size_2D_inference != 0) or (152 % batch_size_2D_inference != 0):
            print("invalid batch size 2D",batch_size_2D_inference)
            raise ValueError("batch_size_2D_inference must be a factor of 192 and 152")
        self.out_c = out_channels
        self.model2D_2.denoise_fn.eval()
        self.model2D_3.denoise_fn.eval()
        self.model_cond2_2.denoise_fn.eval()
        self.model_cond2_3.denoise_fn.eval()

        for param in self.model2D_2.denoise_fn.parameters():
            param.requires_grad = False
        for param in self.model2D_3.denoise_fn.parameters():
            param.requires_grad = False
        for param in self.model_cond2_2.denoise_fn.parameters():
            param.requires_grad = False
        for param in self.model_cond2_3.denoise_fn.parameters():
            param.requires_grad = False
            
            
        # This only works for pl.lightning.LightningModule
        # self.model2D_2.denoise_fn = self.model2D_2.denoise_fn.eval()
        # self.model2D_3.denoise_fn.train = disabled_train
        # self.model2D_3.denoise_fn = self.model2D_3.denoise_fn.eval()
        # self.model2D_3.denoise_fn.train = disabled_train
    def debug_viz(self, arr, name = "debug/viz"):
        # import matplotlib.pyplot as plt
        # nor arr to 0-1
        print("saving img in ", name)
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = arr.cpu().numpy()
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                plt.imshow(arr[i,j,:,:])
                plt.savefig(f"{name}_{i}_{j}.png")
    
    def debug_viz_3D(self, arr, name = "debug/viz"):
        # import matplotlib.pyplot as plt
        # nor arr to 0-1
        print("saving img in ", name)
        # if len(shape) is not 5, we need to add dimension until it's 5:
        if len(arr.shape) == 3:
            arr = arr.unsqueeze(1)
        if len(arr.shape) == 4:
            arr = arr.unsqueeze(1)

        center = arr.shape[2] // 2
        # arr = arr.cpu().numpy()
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                cur_arr = arr[i,j,center,:,:]
                cur_arr = (cur_arr - cur_arr.min()) / (cur_arr.max() - cur_arr.min())
                cur_arr = cur_arr.cpu().numpy()
                plt.imshow(cur_arr)
                plt.savefig(f"{name}_{i}_{j}.png")
                
    def forward(self, x, timesteps):
        
        #  x shape is (batch_size, c, 152, 192, 192) 
        # print("x shape", x.shape) 7: noisy x, cond1(SR), cond2(MT), position, none zero,
        x_cond1, x_cond2 = torch.cat((x[:,0:2], x[:,3:]),dim=1), torch.cat((x[:,0:1], x[:,2:]),dim=1)
        x_weight = extract(self.sqrt_alphas_cumprod, torch.tensor(timesteps, dtype=torch.int64), x.shape)
        # this is necessary because we use residuall training for second model too
        x_cond2[:,0] = x_cond2[:,0] + x_weight * (x_cond1[:,1] - x_cond2[:,1])/2 
        # if timesteps < 100:
        #     self.debug_viz_3D(x_cond1, f"debug/x_cond_1")
        #     self.debug_viz_3D(x_cond2, f"debug/x_cond_2")
        #     raise
        b, c, d, h, w = x_cond1.shape
        b_2D = self.batch_size_2D_inference
        # print(timesteps)
        with torch.no_grad():
            if self.baseline == "3D_only":
                # do not need to do 2D inference if we only use 3D model
                pass
            elif not self.baseline == "3D_feature":
                # no feature is calculated here
                x_2D_2 = x_cond1.permute(0, 3, 1, 2, 4).reshape(-1, c, d, w)
                t_2D = torch.full((b*h,), int(timesteps.item()/self.time_step*self.ntime_steps_2D), device=x_2D_2.device, dtype=torch.long)

                out_2D_2 = torch.zeros(b*h, self.out_c, d, w, device=x_2D_2.device, dtype=x_2D_2.dtype)
                for i in range(0, b*h, b_2D):
                    out_2D_2[i:i+b_2D] = self.model2D_2.denoise_inference(x_2D_2[i:i+b_2D,:self.out_c], t_2D[i:i+b_2D], y_cond=x_2D_2[i:i+b_2D,self.out_c:]) # 8 1 88 64
                    # save the output of 2D model
                    # self.debug_viz( out_2D_2[i:i+b_2D], f"debug/output_2D_2_{i}")
                out_2D_2 = out_2D_2.reshape(b, h, self.out_c, d, w).permute(0, 2, 3, 1, 4)
                
                x_2D_3 = x_cond1.permute(0, 4, 1, 2, 3).reshape(-1, c, d, h)
                out_2D_3 = torch.zeros(b*w, self.out_c, d, h, device=x_2D_3.device, dtype=x_2D_3.dtype)
                for i in range(0, b*h, b_2D):
                    out_2D_3[i:i+b_2D] = self.model2D_3.denoise_inference(x_2D_3[i:i+b_2D,:self.out_c], t_2D[i:i+b_2D], y_cond=x_2D_3[i:i+b_2D,self.out_c:]) # 8 1 88 64
                out_2D_3 = out_2D_3.reshape(b, w, self.out_c, d, h).permute(0, 2, 3, 4, 1)

                # second condition:
                ############################################################################################################
                ############################################################################################################
                ############################################################################################################
                x_cond2_2 = x_cond2.permute(0, 3, 1, 2, 4).reshape(-1, c, d, w)
                out_cond2_2 = torch.zeros(b*h, self.out_c, d, w, device=x_cond2_2.device, dtype=x_cond2_2.dtype)
                for i in range(0, b*h, b_2D):
                    out_cond2_2[i:i+b_2D] = self.model_cond2_2.denoise_inference(x_cond2_2[i:i+b_2D,:self.out_c], t_2D[i:i+b_2D], y_cond=x_cond2_2[i:i+b_2D,self.out_c:]) # 8 1 88 64
                    # save the output of 2D model
                    # self.debug_viz( out_2D_2[i:i+b_2D], f"debug/output_2D_2_{i}")
                out_cond2_2 = out_cond2_2.reshape(b, h, self.out_c, d, w).permute(0, 2, 3, 1, 4)
                
                x_cond2_3 = x_cond2.permute(0, 4, 1, 2, 3).reshape(-1, c, d, h)
                out_cond2_3 = torch.zeros(b*w, self.out_c, d, h, device=x_cond2_3.device, dtype=x_cond2_3.dtype)
                for i in range(0, b*h, b_2D):
                    out_cond2_3[i:i+b_2D] = self.model_cond2_3.denoise_inference(x_cond2_3[i:i+b_2D,:self.out_c], t_2D[i:i+b_2D], y_cond=x_cond2_3[i:i+b_2D,self.out_c:]) # 8 1 88 64
                out_cond2_3 = out_cond2_3.reshape(b, w, self.out_c, d, h).permute(0, 2, 3, 4, 1)
            
            else: 
                # features are calculated here
                x_2D_2 = x_cond1.permute(0, 3, 1, 2, 4).reshape(-1, c, d, w)
                t_2D = torch.full((b*h,), int(timesteps.item()/self.time_step*self.ntime_steps_2D), device=x_2D_2.device, dtype=torch.long)

                out_2D_2 = torch.zeros(b*h, self.out_c, d, w, device=x_2D_2.device, dtype=x_2D_2.dtype)
                feature_2D_2 = torch.zeros(b*h, self.featue_c, d, w, device=x_2D_2.device, dtype=x_2D_2.dtype)
                for i in range(0, b*h, b_2D):
                    out_2D_2[i:i+b_2D], F = self.model2D_2.denoise_inference(x_2D_2[i:i+b_2D,:self.out_c], t_2D[i:i+b_2D], y_cond=x_2D_2[i:i+b_2D,self.out_c:], feature = True) # 8 1 88 64
                    feature_2D_2[i:i+b_2D] = F[-2]

                out_2D_2 = out_2D_2.reshape(b, h, self.out_c, d, w).permute(0, 2, 3, 1, 4)
                feature_2D_2 = feature_2D_2.reshape(b, h, self.featue_c, d, w).permute(0, 2, 3, 1, 4)

                x_2D_3 = x_cond1.permute(0, 4, 1, 2, 3).reshape(-1, c, d, h)
                out_2D_3 = torch.zeros(b*w, self.out_c, d, h, device=x_2D_3.device, dtype=x_2D_3.dtype)
                feature_2D_3 = torch.zeros(b*w, self.featue_c, d, h, device=x_2D_3.device, dtype=x_2D_3.dtype)
                for i in range(0, b*h, b_2D):
                    out_2D_3[i:i+b_2D], F = self.model2D_3.denoise_inference(x_2D_3[i:i+b_2D,:self.out_c], t_2D[i:i+b_2D], y_cond=x_2D_3[i:i+b_2D,self.out_c:], feature = True) # 8 1 88 64
                    feature_2D_3[i:i+b_2D] = F[-1]

                out_2D_3 = out_2D_3.reshape(b, w, self.out_c, d, h).permute(0, 2, 3, 4, 1)
                feature_2D_3 = feature_2D_3.reshape(b, w, self.featue_c, d, h).permute(0, 2, 3, 4, 1)
                
                # second condition:
                ############################################################################################################
                ############################################################################################################
                ############################################################################################################
                x_cond2_2 = x_cond2.permute(0, 3, 1, 2, 4).reshape(-1, c, d, w)
                out_cond2_2 = torch.zeros(b*h, self.out_c, d, w, device=x_cond2_2.device, dtype=x_cond2_2.dtype)
                feature_cond2_2 = torch.zeros(b*h, self.featue_c, d, w, device=x_cond2_2.device, dtype=x_cond2_2.dtype)
                for i in range(0, b*h, b_2D):
                    out_cond2_2[i:i+b_2D], F = self.model_cond2_2.denoise_inference(x_cond2_2[i:i+b_2D,:self.out_c], t_2D[i:i+b_2D], y_cond=x_cond2_2[i:i+b_2D,self.out_c:], feature = True) # 8 1 88 64
                    feature_cond2_2[i:i+b_2D] = F[-2]

                out_cond2_2 = out_cond2_2.reshape(b, h, self.out_c, d, w).permute(0, 2, 3, 1, 4)
                feature_cond2_2 = feature_cond2_2.reshape(b, h, self.featue_c, d, w).permute(0, 2, 3, 1, 4)

                x_cond2_3 = x_cond2.permute(0, 4, 1, 2, 3).reshape(-1, c, d, h)
                out_cond2_3 = torch.zeros(b*w, self.out_c, d, h, device=x_cond2_3.device, dtype=x_cond2_3.dtype)
                feature_cond2_3 = torch.zeros(b*w, self.featue_c, d, h, device=x_cond2_3.device, dtype=x_cond2_3.dtype)
                for i in range(0, b*h, b_2D):
                    out_cond2_3[i:i+b_2D], F = self.model_cond2_3.denoise_inference(x_cond2_3[i:i+b_2D,:self.out_c], t_2D[i:i+b_2D], y_cond=x_cond2_3[i:i+b_2D,self.out_c:], feature = True) # 8 1 88 64
                    feature_cond2_3[i:i+b_2D] = F[-1]

                out_cond2_3 = out_cond2_3.reshape(b, w, self.out_c, d, h).permute(0, 2, 3, 4, 1)
                feature_cond2_3 = feature_cond2_3.reshape(b, w, self.featue_c, d, h).permute(0, 2, 3, 4, 1)
                

        if self.baseline == "2D":
            w2 = 0.0
            w3 = 1.0
            return w2 * out_2D_2 + w3 * out_2D_3
        elif self.baseline == "merge":
            w2 = 0.25
            w3 = 0.25
            w4 = 0.25
            w5 = 0.25 #0.25
            return w2 * out_2D_2 + w3 * out_2D_3 + w4 * out_cond2_2 + w5 * out_cond2_3
        elif self.baseline == "TPDM":
            w2 = 0.25
            w3 = 0.5
            w4 = 0.75
            # randomly pick one of the 2D model to use
            if np.random.rand() < w2:
                return out_2D_2
            elif np.random.rand() < w3:
                return out_2D_3
            elif np.random.rand() < w4:
                return out_cond2_2
            else:
                return out_cond2_3
            
        elif self.baseline == "3D":
            # inference for 3D model
            x_3D = torch.cat([x, out_2D_2.detach(), out_2D_3.detach(), out_cond2_2.detach(), out_cond2_3.detach()], dim = 1)
            out_3D = self.model_3D(x_3D, timesteps)
            return out_3D
        elif self.baseline == "3D_only":
            # inference for 3D model only
            out_3D = self.model_3D(x, timesteps)
            return out_3D
        elif self.baseline == "3D_feature":
            # inference for 3D model
            # print("x", x.shape)
            # print(out_2D_2.shape, out_2D_3.shape, out_cond2_2.shape, out_cond2_3.shape)
            x_3D = torch.cat([x, out_2D_2.detach(), out_2D_3.detach(), out_cond2_2.detach(), out_cond2_3.detach()], dim = 1)
            feature_3D = torch.cat([feature_2D_2.detach(), feature_2D_3.detach(), feature_cond2_2.detach(), feature_cond2_3.detach()], dim = 1)
            
            model_ensemble = self.model_3D(x_3D, timesteps, feature_2D = feature_3D)
            # print(model_ensemble.shape, x_3D.shape)
            return model_ensemble[:,0] * x_3D[:,-4] + model_ensemble[:,1] * x_3D[:,-3] + model_ensemble[:,2] * x_3D[:,-2] + model_ensemble[:,3] * x_3D[:,-1] 
            # return model_ensemble[:,0] * x_3D[:,-4] + model_ensemble[:,1] * x_3D[:,-3] + model_ensemble[:,2] * x_3D[:,-2] + model_ensemble[:,3] * x_3D[:,-1] 
        else:
            raise ValueError("baseline must be 2D, 3D or merge")


class model_ensemble_three_model(nn.Module):
    # baseline for two and a half order
    def __init__(self, model_3D, model2D_1, model2D_2, model2D_3,
                 batch_size_2D_inference = 8, 
                 time_step = 1000, 
                 out_channels = 1, 
                 ntime_steps_2D = 1000, 
                 baseline = "3D",
                 weight_3D = 1.0
                 ):
        super(model_ensemble_three_model, self).__init__()
        self.ntime_steps_2D = ntime_steps_2D
        self.model_3D = model_3D
        self.model2D_1 = model2D_1.netG
        self.model2D_2 = model2D_2.netG
        self.model2D_3 = model2D_3.netG
        self.baseline = baseline
        # TODO change this from hard coding
        self.featue_c = 64
        self.weight_3D = weight_3D
        # if model2D_3 is a DDP model, we need to access the module
        
        if hasattr(self.model2D_3, "module"):
            self.model2D_3 = self.model2D_3.module
            self.model2D_2 = self.model2D_2.module
            self.model2D_1 = self.model2D_1.module
            
        self.batch_size_2D_inference = batch_size_2D_inference
        self.time_step = time_step
        if (192 % batch_size_2D_inference != 0) or (152 % batch_size_2D_inference != 0):
            print("invalid batch size 2D", batch_size_2D_inference)
            raise ValueError("batch_size_2D_inference must be a factor of 192 and 152")
        self.out_c = out_channels

        self.model2D_1.denoise_fn.eval()
        self.model2D_2.denoise_fn.eval()
        self.model2D_3.denoise_fn.eval()

        for param in self.model2D_1.denoise_fn.parameters():
            param.requires_grad = False
        for param in self.model2D_2.denoise_fn.parameters():
            param.requires_grad = False
        for param in self.model2D_3.denoise_fn.parameters():
            param.requires_grad = False

        if baseline == "3D_feature":
            # self.feature_layers = []
            # zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1, padding_mode = 'replicate'))
            self.feature_layer_1_1 = zero_module(nn.Conv3d(512, 192, (8,1,1), stride = (8,1,1)))
            self.feature_layer_1_4 = zero_module(nn.Conv3d(256, 192, (4,1,1), stride = (4,1,1)))
            self.feature_layer_1_7 = zero_module(nn.Conv3d(128, 128, (2,1,1), stride = (2,1,1)))
            
            self.feature_layer_2_1 = zero_module(nn.Conv3d(512, 192, (1,8,1), stride = (1,8,1)))
            self.feature_layer_2_4 = zero_module(nn.Conv3d(256, 192, (1,4,1), stride = (1,4,1)))
            self.feature_layer_2_7 = zero_module(nn.Conv3d(128, 128, (1,2,1), stride = (1,2,1)))
            
            self.feature_layer_3_1 = zero_module(nn.Conv3d(512, 192, (1,1,8), stride = (1,1,8)))
            self.feature_layer_3_4 = zero_module(nn.Conv3d(256, 192, (1,1,4), stride = (1,1,4)))
            self.feature_layer_3_7 = zero_module(nn.Conv3d(128, 128, (1,1,2), stride = (1,1,2)))
            

            self.feature_extract_layer_1_10 = zero_module(nn.Conv3d(64, 64, 1, stride = 1, padding = 0))
            self.feature_extract_layer_2_10 = zero_module(nn.Conv3d(64, 64, 1, stride = 1, padding = 0))
            self.feature_extract_layer_3_10 = zero_module(nn.Conv3d(64, 64, 1, stride = 1, padding = 0))

        else:
            self.feature_layers = None
        # This only works for pl.lightning.LightningModule
        # self.model2D_2.denoise_fn = self.model2D_2.denoise_fn.eval()
        # self.model2D_3.denoise_fn.train = disabled_train
        # self.model2D_3.denoise_fn = self.model2D_3.denoise_fn.eval()
        # self.model2D_3.denoise_fn.train = disabled_train
    def debug_viz(self, arr, name = "debug/viz"):
        # import matplotlib.pyplot as plt
        # nor arr to 0-1
        print("saving img in ", name)
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = arr.cpu().numpy()
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                plt.imshow(arr[i,j,:,:])
                plt.savefig(f"{name}_{i}_{j}.png")

    def debug_viz_3D(self, arr, name = "debug/viz"):
        # import matplotlib.pyplot as plt
        # nor arr to 0-1
        print("saving img in ", name)
        # if len(shape) is not 5, we need to add dimension until it's 5:
        if len(arr.shape) == 3:
            arr = arr.unsqueeze(1)
        if len(arr.shape) == 4:
            arr = arr.unsqueeze(1)

        center = arr.shape[2] // 2
        # arr = arr.cpu().numpy()
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                cur_arr = arr[i,j,center,:,:]
                cur_arr = (cur_arr - cur_arr.min()) / (cur_arr.max() - cur_arr.min())
                cur_arr = cur_arr.cpu().numpy()
                plt.imshow(cur_arr)
                plt.savefig(f"{name}_{i}_{j}.png")
                
    def forward(self, x, timesteps):
        # x shape is (batch_size, c, 152, 192, 192)
        # print("debugging t for 3D:", timesteps)
        b, c, d, h, w = x.shape
        b_2D = self.batch_size_2D_inference
        # print(timesteps)
        if self.baseline == "dummy":
            return x[:,0:1]
        with torch.no_grad():
            if self.baseline == "3D_only":
                # do not need to do 2D inference if we only use 3D model
                pass
            elif not self.baseline == "3D_feature":
                # no feature is calculated here
                x_2D_2 = x.permute(0, 3, 1, 2, 4).reshape(-1, c, d, w)
                t_2D = torch.full((b*h,), int(timesteps.item()/self.time_step*self.ntime_steps_2D), device=x_2D_2.device, dtype=torch.long)

                out_2D_2 = torch.zeros(b*h, self.out_c, d, w, device=x_2D_2.device, dtype=x_2D_2.dtype)
                for i in range(0, b*h, b_2D):
                    out_2D_2[i:i+b_2D] = self.model2D_2.denoise_inference(x_2D_2[i:i+b_2D,:self.out_c], t_2D[i:i+b_2D], y_cond=x_2D_2[i:i+b_2D,self.out_c:]) # 8 1 88 64
                    # save the output of 2D model
                    # self.debug_viz( out_2D_2[i:i+b_2D], f"debug/output_2D_2_{i}")
                out_2D_2 = out_2D_2.reshape(b, h, self.out_c, d, w).permute(0, 2, 3, 1, 4)
                
                x_2D_3 = x.permute(0, 4, 1, 2, 3).reshape(-1, c, d, h)
                out_2D_3 = torch.zeros(b*w, self.out_c, d, h, device=x_2D_3.device, dtype=x_2D_3.dtype)
                for i in range(0, b*h, b_2D):
                    out_2D_3[i:i+b_2D] = self.model2D_3.denoise_inference(x_2D_3[i:i+b_2D,:self.out_c], t_2D[i:i+b_2D], y_cond=x_2D_3[i:i+b_2D,self.out_c:]) # 8 1 88 64
                out_2D_3 = out_2D_3.reshape(b, w, self.out_c, d, h).permute(0, 2, 3, 4, 1)

                x_2D_1 = x.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
                t_2D_1 = torch.full((b*d,), int(timesteps.item()/self.time_step*self.ntime_steps_2D), device=x_2D_2.device, dtype=torch.long)
                out_2D_1 = torch.zeros(b*d, self.out_c, h, w, device=x_2D_1.device, dtype=x_2D_1.dtype)
                for i in range(0, b*d, b_2D):
                    out_2D_1[i:i+b_2D] = self.model2D_1.denoise_inference(x_2D_1[i:i+b_2D,:self.out_c], t_2D_1[i:i+b_2D], y_cond=x_2D_1[i:i+b_2D,self.out_c:])
                out_2D_1 = out_2D_1.reshape(b, d, self.out_c, h, w).permute(0, 2, 1, 3, 4)
            else: 
                # raise NotImplementedError("3D_feature is not supported for this basesline that use three 2D models")
                # 2D_features are calculated here
                x_2D_2 = x.permute(0, 3, 1, 2, 4).reshape(-1, c, d, w)
                t_2D = torch.full((b*h,), int(timesteps.item()/self.time_step*self.ntime_steps_2D), device=x_2D_2.device, dtype=torch.long)
                                    
                out_2D_2 = torch.zeros(b*h, self.out_c, d, w, device=x_2D_2.device, dtype=x_2D_2.dtype)
                out_feature_2_1 = torch.zeros(b*h, 512, 19, 24, device=x_2D_2.device, dtype=x_2D_2.dtype)
                out_feature_2_4 = torch.zeros(b*h, 256, 38, 48, device=x_2D_2.device, dtype=x_2D_2.dtype)
                out_feature_2_7 = torch.zeros(b*h, 128, 76, 96, device=x_2D_2.device, dtype=x_2D_2.dtype)
                out_feature_2_10 = torch.zeros(b*h, 64, 152, 192, device=x_2D_2.device, dtype=x_2D_2.dtype)
                # feature_2D_2 = torch.zeros(b*h, self.featue_c, d, w, device=x_2D_2.device, dtype=x_2D_2.dtype)
                for i in range(0, b*h, b_2D):
                    out_2D_2[i:i+b_2D], F = self.model2D_2.denoise_inference(x_2D_2[i:i+b_2D,:self.out_c], t_2D[i:i+b_2D], y_cond=x_2D_2[i:i+b_2D,self.out_c:], feature = True) # 8 1 88 64
                    # feature_2D_2[i:i+b_2D] = F[-2]
                    out_feature_2_1[i:i+b_2D] = F[1]
                    out_feature_2_4[i:i+b_2D] = F[4]
                    out_feature_2_7[i:i+b_2D] = F[7]
                    out_feature_2_10[i:i+b_2D] = F[10]

                out_2D_2 = out_2D_2.reshape(b, h, self.out_c, d, w).permute(0, 2, 3, 1, 4)
                # feature_2D_2 = feature_2D_2.reshape(b, h, self.featue_c, d, w).permute(0, 2, 3, 1, 4)
                out_feature_2_1 = out_feature_2_1.reshape(b, h, 512, 19, 24).permute(0, 2, 3, 1, 4)
                out_feature_2_4 = out_feature_2_4.reshape(b, h, 256, 38, 48).permute(0, 2, 3, 1, 4)
                out_feature_2_7 = out_feature_2_7.reshape(b, h, 128, 76, 96).permute(0, 2, 3, 1, 4)
                out_feature_2_10 = out_feature_2_10.reshape(b, h, 64, 152, 192).permute(0, 2, 3, 1, 4)

                x_2D_3 = x.permute(0, 4, 1, 2, 3).reshape(-1, c, d, h)
                out_2D_3 = torch.zeros(b*w, self.out_c, d, h, device=x_2D_3.device, dtype=x_2D_3.dtype)
                # feature_2D_3 = torch.zeros(b*w, self.featue_c, d, h, device=x_2D_3.device, dtype=x_2D_3.dtype)
                out_feature_3_1 = torch.zeros(b*w, 512, 19, 24, device=x_2D_2.device, dtype=x_2D_2.dtype)
                out_feature_3_4 = torch.zeros(b*w, 256, 38, 48, device=x_2D_2.device, dtype=x_2D_2.dtype)
                out_feature_3_7 = torch.zeros(b*w, 128, 76, 96, device=x_2D_2.device, dtype=x_2D_2.dtype)
                out_feature_3_10 = torch.zeros(b*w, 64, 152, 192, device=x_2D_2.device, dtype=x_2D_2.dtype)
                
                for i in range(0, b*h, b_2D):
                    out_2D_3[i:i+b_2D], F = self.model2D_3.denoise_inference(x_2D_3[i:i+b_2D,:self.out_c], t_2D[i:i+b_2D], y_cond=x_2D_3[i:i+b_2D,self.out_c:], feature = True) # 8 1 88 64
                    # feature_2D_3[i:i+b_2D] = F[-1]
                    out_feature_3_1[i:i+b_2D] = F[1]
                    out_feature_3_4[i:i+b_2D] = F[4]
                    out_feature_3_7[i:i+b_2D] = F[7]
                    out_feature_3_10[i:i+b_2D] = F[10]

                out_2D_3 = out_2D_3.reshape(b, w, self.out_c, d, h).permute(0, 2, 3, 4, 1)
                # feature_2D_3 = feature_2D_3.reshape(b, w, self.featue_c, d, h).permute(0, 2, 3, 4, 1)
                out_feature_3_1 = out_feature_3_1.reshape(b, w, 512, 19, 24).permute(0, 2, 3, 4, 1)
                out_feature_3_4 = out_feature_3_4.reshape(b, w, 256, 38, 48).permute(0, 2, 3, 4, 1)
                out_feature_3_7 = out_feature_3_7.reshape(b, w, 128, 76, 96).permute(0, 2, 3, 4, 1)
                out_feature_3_10 = out_feature_3_10.reshape(b, w, 64, 152, 192).permute(0, 2, 3, 4, 1)
                
                x_2D_1 = x.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
                t_2D_1 = torch.full((b*d,), int(timesteps.item()/self.time_step*self.ntime_steps_2D), device=x_2D_2.device, dtype=torch.long)
                out_2D_1 = torch.zeros(b*d, self.out_c, h, w, device=x_2D_1.device, dtype=x_2D_1.dtype)
                out_feature_1_1 = torch.zeros(b*d, 512, 24, 24, device=x_2D_2.device, dtype=x_2D_2.dtype)
                out_feature_1_4 = torch.zeros(b*d, 256, 48, 48, device=x_2D_2.device, dtype=x_2D_2.dtype)
                out_feature_1_7 = torch.zeros(b*d, 128, 96, 96, device=x_2D_2.device, dtype=x_2D_2.dtype)
                out_feature_1_10 = torch.zeros(b*d, 64, 192, 192, device=x_2D_2.device, dtype=x_2D_2.dtype)
                
                for i in range(0, b*d, b_2D):
                    out_2D_1[i:i+b_2D], F = self.model2D_1.denoise_inference(x_2D_1[i:i+b_2D,:self.out_c], t_2D_1[i:i+b_2D], y_cond=x_2D_1[i:i+b_2D,self.out_c:], feature = True)
                    out_feature_1_1[i:i+b_2D] = F[1]
                    out_feature_1_4[i:i+b_2D] = F[4]
                    out_feature_1_7[i:i+b_2D] = F[7]
                    out_feature_1_10[i:i+b_2D] = F[10]
                    
                out_2D_1 = out_2D_1.reshape(b, d, self.out_c, h, w).permute(0, 2, 1, 3, 4)
                out_feature_1_1 = out_feature_1_1.reshape(b, d, 512, 24, 24).permute(0, 2, 1, 3, 4)
                out_feature_1_4 = out_feature_1_4.reshape(b, d, 256, 48, 48).permute(0, 2, 1, 3, 4)
                out_feature_1_7 = out_feature_1_7.reshape(b, d, 128, 96, 96).permute(0, 2, 1, 3, 4)
                out_feature_1_10 = out_feature_1_10.reshape(b, d, 64, 192, 192).permute(0, 2, 1, 3, 4)

        if self.baseline == "2D":
            w2 = 0.0
            w3 = 1.0
            return w2 * out_2D_2 + w3 * out_2D_3
        elif self.baseline == "merge":
            w1 = 1.0 / 3
            w2 = 1.0 / 3
            w3 = 1.0 / 3
            return w1 * out_2D_1 + w2 * out_2D_2 + w3 * out_2D_3
        elif self.baseline == "TPDM":
            w2 = 0.1
            w3 = 0.666666
            # randomly pick one of the 2D model to use
            if np.random.rand() < w2:
                return out_2D_1
            elif np.random.rand() < w3:
                return out_2D_2
            else:
                return out_2D_3
            
        elif self.baseline == "3D":
            # inference for 3D model
            x_3D = torch.cat([x, out_2D_1.detach(), out_2D_2.detach(), out_2D_3.detach()], dim = 1)
            ensemble_w = self.model_3D(x_3D, timesteps)
            out_3D = (ensemble_w[:,0] + 0.3334) * out_2D_1 + (ensemble_w[:,1]+0.3333) * out_2D_2 +  (0.3333 - ensemble_w[:,0] - ensemble_w[:,1]) * out_2D_3 + ensemble_w[:,2]
            return out_3D
        elif self.baseline == "3D_only":
            # inference for 3D model only
            out_3D = self.model_3D(x, timesteps)
            return out_3D
        elif self.baseline == "3D_feature":
            # inference for 3D model
            out_feature_1_1 = self.feature_layer_1_1(out_feature_1_1)
            out_feature_1_4 = self.feature_layer_1_4(out_feature_1_4)
            out_feature_1_7 = self.feature_layer_1_7(out_feature_1_7)
                
            out_feature_2_1 = self.feature_layer_2_1(out_feature_2_1)
            out_feature_2_4 = self.feature_layer_2_4(out_feature_2_4)
            out_feature_2_7 = self.feature_layer_2_7(out_feature_2_7)
            
            out_feature_3_1 = self.feature_layer_3_1(out_feature_3_1)
            out_feature_3_4 = self.feature_layer_3_4(out_feature_3_4)
            out_feature_3_7 = self.feature_layer_3_7(out_feature_3_7)
            
            out_feature_1 = out_feature_1_1 + out_feature_2_1 + out_feature_3_1
            out_feature_4 = out_feature_1_4 + out_feature_2_4 + out_feature_3_4
            out_feature_7 = out_feature_1_7 + out_feature_2_7 + out_feature_3_7
            out_feature_10 = self.feature_extract_layer_1_10(out_feature_1_10) + self.feature_extract_layer_2_10(out_feature_2_10) + self.feature_extract_layer_3_10(out_feature_3_10)
            out_features = [out_feature_1,out_feature_4,out_feature_7,out_feature_10]
        
            x_3D = torch.cat([x, out_2D_1.detach(), out_2D_2.detach(), out_2D_3.detach()], dim = 1)
            ensemble_w = self.model_3D(x_3D, timesteps, feature_2D = out_features)
            out_3D = (ensemble_w[:,0] + 0.3334) * out_2D_1 + (ensemble_w[:,1]+0.3333) * out_2D_2 +  (0.3333 - ensemble_w[:,0] - ensemble_w[:,1]) * out_2D_3 + self.weight_3D * ensemble_w[:,2]
            return out_3D
        
            # x_3D = torch.cat([x, out_2D_1.detach(), out_2D_2.detach(), out_2D_3.detach()], dim = 1)
            # # feature_3D = torch.cat([feature_2D_2.detach(), feature_2D_3.detach()], dim = 1)
            # # x_3D_final = torch.cat([x_3D], dim = 1)
            
            # ensemble_w = self.model_3D(x_3D, timesteps, feature_2D = feature_3D)
            # out_3D = (ensemble_w[:,0] + 0.3334) * out_2D_1 + (ensemble_w[:,1]+0.3333) * out_2D_2 +  (0.3333 - ensemble_w[:,0] - ensemble_w[:,1]) * out_2D_3 + ensemble_w[:,2]
            # return out_3D
        else:
            raise ValueError("baseline must be 2D, 3D or merge")


class model_ensemble_three_25_model(nn.Module):
    # baseline for two and a half order
    def __init__(self, model_3D, model2D_1, model2D_2, model2D_3,
                 batch_size_2D_inference = 8, 
                 time_step = 1000, 
                 out_channels = 1, 
                 ntime_steps_2D = 1000, 
                 baseline = "3D",
                 nearby_slice = 2,
                 weight_3D = 1.0
                 ):
        super(model_ensemble_three_25_model, self).__init__()
        self.ntime_steps_2D = ntime_steps_2D
        self.model_3D = model_3D
        self.model2D_1 = model2D_1.netG
        self.model2D_2 = model2D_2.netG
        self.model2D_3 = model2D_3.netG
        self.baseline = baseline
        self.nearby_slice = nearby_slice
        self.weight_3D = weight_3D
        if baseline == "3D_feature":
            # self.feature_layers = []
            # zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1, padding_mode = 'replicate'))
            self.feature_layer_1_1 = zero_module(nn.Conv3d(512, 192, (8,1,1), stride = (8,1,1)))
            self.feature_layer_1_4 = zero_module(nn.Conv3d(256, 192, (4,1,1), stride = (4,1,1)))
            self.feature_layer_1_7 = zero_module(nn.Conv3d(128, 128, (2,1,1), stride = (2,1,1)))
            
            self.feature_layer_2_1 = zero_module(nn.Conv3d(512, 192, (1,8,1), stride = (1,8,1)))
            self.feature_layer_2_4 = zero_module(nn.Conv3d(256, 192, (1,4,1), stride = (1,4,1)))
            self.feature_layer_2_7 = zero_module(nn.Conv3d(128, 128, (1,2,1), stride = (1,2,1)))
            
            self.feature_layer_3_1 = zero_module(nn.Conv3d(512, 192, (1,1,8), stride = (1,1,8)))
            self.feature_layer_3_4 = zero_module(nn.Conv3d(256, 192, (1,1,4), stride = (1,1,4)))
            self.feature_layer_3_7 = zero_module(nn.Conv3d(128, 128, (1,1,2), stride = (1,1,2)))
            
            # learning layers, this makes it possible to just sum them up!
            # self.feature_extract_layer_1_1 = zero_module(nn.Conv3d(192, 192, 3, stride = 1, padding = 1))
            # self.feature_extract_layer_1_4 = zero_module(nn.Conv3d(128, 128, 3, stride = 1, padding = 1))
            # self.feature_extract_layer_1_7 = zero_module(nn.Conv3d(128, 128, 3, stride = 1, padding = 1))
            self.feature_extract_layer_1_10 = zero_module(nn.Conv3d(64, 64, 1, stride = 1, padding = 0))
            
            # self.feature_extract_layer_2_1 = zero_module(nn.Conv3d(256, 192, 3, stride = 1, padding = 1))
            # self.feature_extract_layer_2_4 = zero_module(nn.Conv3d(256, 256, 3, stride = 1, padding = 1))
            # self.feature_extract_layer_2_7 = zero_module(nn.Conv3d(128, 128, 3, stride = 1, padding = 1))
            self.feature_extract_layer_2_10 = zero_module(nn.Conv3d(64, 64, 1, stride = 1, padding = 0))
            
            # self.feature_extract_layer_3_1 = zero_module(nn.Conv3d(256, 192, 3, stride = 1, padding = 1))
            # self.feature_extract_layer_3_4 = zero_module(nn.Conv3d(256, 256, 3, stride = 1, padding = 1))
            # self.feature_extract_layer_3_7 = zero_module(nn.Conv3d(128, 128, 3, stride = 1, padding = 1))
            self.feature_extract_layer_3_10 = zero_module(nn.Conv3d(64, 64, 1, stride = 1, padding = 0))

        else:
            self.feature_layers = None
        assert nearby_slice == 2
        # Currently only support nearby_slice = 2
        # TODO change this from hard coding
        self.featue_c = 64
        # if model2D_3 is a DDP model, we need to access the module
        if hasattr(self.model2D_3, "module"):
            self.model2D_3 = self.model2D_3.module
            self.model2D_2 = self.model2D_2.module
            self.model2D_1 = self.model2D_1.module
            
        self.batch_size_2D_inference = batch_size_2D_inference
        self.time_step = time_step
        if (192 % batch_size_2D_inference != 0) or (152 % batch_size_2D_inference != 0):
            print("invalid batch size 2D", batch_size_2D_inference)
            raise ValueError("batch_size_2D_inference must be a factor of 192 and 152")
        self.out_c = out_channels

        self.model2D_1.denoise_fn.eval()
        self.model2D_2.denoise_fn.eval()
        self.model2D_3.denoise_fn.eval()

        for param in self.model2D_1.denoise_fn.parameters():
            param.requires_grad = False
        for param in self.model2D_2.denoise_fn.parameters():
            param.requires_grad = False
        for param in self.model2D_3.denoise_fn.parameters():
            param.requires_grad = False
        # This only works for pl.lightning.LightningModule
        # self.model2D_2.denoise_fn = self.model2D_2.denoise_fn.eval()
        # self.model2D_3.denoise_fn.train = disabled_train
        # self.model2D_3.denoise_fn = self.model2D_3.denoise_fn.eval()
        # self.model2D_3.denoise_fn.train = disabled_train
    def debug_viz(self, arr, name = "debug/viz"):
        # import matplotlib.pyplot as plt
        # nor arr to 0-1
        print("saving img in ", name)
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = arr.cpu().numpy()
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                plt.imshow(arr[i,j,:,:])
                plt.savefig(f"{name}_{i}_{j}.png")

    def debug_viz_3D(self, arr, name = "debug/viz"):
        # import matplotlib.pyplot as plt
        # nor arr to 0-1
        print("saving img in ", name)
        # if len(shape) is not 5, we need to add dimension until it's 5:
        if len(arr.shape) == 3:
            arr = arr.unsqueeze(1)
        if len(arr.shape) == 4:
            arr = arr.unsqueeze(1)

        center = arr.shape[2] // 2
        # arr = arr.cpu().numpy()
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                cur_arr = arr[i,j,center,:,:]
                cur_arr = (cur_arr - cur_arr.min()) / (cur_arr.max() - cur_arr.min())
                cur_arr = cur_arr.cpu().numpy()
                plt.imshow(cur_arr)
                plt.savefig(f"{name}_{i}_{j}.png")
    
    def slicing(self, arr, idx):
        # return the idx-th slice of the tensor
        # return all zeros if idx is out of bound
        if idx < 0:
            return arr[0]
        elif idx >= arr.shape[0]:
            return arr[-1]
        else :
            return arr[idx]
        
    def forward(self, x, timesteps):
        # x shape is (batch_size, c, 152, 192, 192)
        # print("debugging t for 3D:", timesteps)
        b, c, d, h, w = x.shape
        b_2D = self.batch_size_2D_inference
        # print(timesteps)
        if self.baseline == "dummy":
            return x[:,0:1]
        with torch.no_grad():
            if self.baseline == "3D_only":
                # do not need to do 2D inference if we only use 3D model
                # Please use model ensembler for 2 models for this option
                raise NotImplementedError
                # pass
            elif not self.baseline == "3D_feature":
                # Here the random crop size in all directionns have to be divisible by 2D batch size
                # out_feature_3D_1 = 
                
                x_2D_2 = x.permute(0, 3, 1, 2, 4).reshape(-1, c, d, w)
                t_2D = torch.full((b*h,), int(timesteps.item()/self.time_step*self.ntime_steps_2D), device=x_2D_2.device, dtype=torch.long)
                
                # no feature is calculated here
                input_c = x_2D_2.shape[1] - self.out_c
                img_idx = input_c - 3 - 1
                num_modality = img_idx # // 5
                # print("num_modality", num_modality)
                # channel_idx = list(range(img_idx)) # data
                # channel_idx += [img_idx+2,img_idx+7,img_idx+12,img_idx+17] # pos emb and none zero mask 
                out_2D_2 = torch.zeros(b*h, self.out_c, d, w, device=x_2D_2.device, dtype=x_2D_2.dtype)
                # print("x_2D_2 shape", x_2D_2.shape): 64, 6, 88, 64]
                for i in range(0, b*h, b_2D):
                    total_cond_array = []
                    for idx in range(i, i+b_2D):
                        # needs to be changed when batch size is not 1
                        # x_2D_2[i:i+b_2D, self.out_c:]
                        data_array = []
                        for nearby_idx in range(-2,3):
                            data_array.append(self.slicing(x_2D_2, idx + nearby_idx)[self.out_c:self.out_c+num_modality])
                        data_array = torch.cat(data_array, dim = 0) # 5 x 1,88,64 -> 5,88,64 
                        cond_array = torch.cat([data_array,x_2D_2[idx,self.out_c+num_modality:]], dim = 0) # 5,88,64 + 4,88,64 -> 9,88,64
                        total_cond_array.append(cond_array)
                    total_cond_array = torch.stack(total_cond_array) # 8 x 9, 88, 64 -> 8, 9, 88, 64
                    # print(total_cond_array.shape) # 8, 9, 88, 64 (152 192)
                    # self.debug_viz(total_cond_array[0:1], f"debug/output_2D_2")
                    # raise
                    # print(x_2D_2[i:i+b_2D,:self.out_c].shape, total_cond_array.shape) # 8, 1/9, 88, 64 / (152,192)
                    out_2D_2[i:i+b_2D] = self.model2D_2.denoise_inference(x_2D_2[i:i+b_2D,:self.out_c], t_2D[i:i+b_2D], y_cond=total_cond_array) # 8 1 88 64
                    # save the output of 2D model
                    # self.debug_viz( out_2D_2[i:i+b_2D], f"debug/output_2D_2_{i}")
                out_2D_2 = out_2D_2.reshape(b, h, self.out_c, d, w).permute(0, 2, 3, 1, 4)
                
                x_2D_3 = x.permute(0, 4, 1, 2, 3).reshape(-1, c, d, h)
                out_2D_3 = torch.zeros(b*w, self.out_c, d, h, device=x_2D_3.device, dtype=x_2D_3.dtype)
                for i in range(0, b*h, b_2D):
                    # out_2D_3[i:i+b_2D] = self.model2D_3.denoise_inference(x_2D_3[i:i+b_2D,:self.out_c], t_2D[i:i+b_2D], y_cond=x_2D_3[i:i+b_2D,self.out_c:]) # 8 1 88 64
                    total_cond_array = []
                    for idx in range(i, i+b_2D):
                        data_array = []
                        for nearby_idx in range(-2,3):
                            data_array.append(self.slicing(x_2D_3, idx + nearby_idx)[self.out_c:self.out_c+num_modality])
                        data_array = torch.cat(data_array, dim = 0)
                        cond_array = torch.cat([data_array,x_2D_3[idx,self.out_c+num_modality:]], dim = 0)
                        total_cond_array.append(cond_array)
                    total_cond_array = torch.stack(total_cond_array)
                    # import pdb; pdb.set_trace()
                    # print(total_cond_array.shape) # 8, 9, 88, 64
                    # self.debug_viz(total_cond_array[0:1], f"debug/output_2D_3")
                    # raise
                    out_2D_3[i:i+b_2D] = self.model2D_2.denoise_inference(x_2D_3[i:i+b_2D,:self.out_c], t_2D[i:i+b_2D], y_cond=total_cond_array) # 8 1 88 64
                    # save the output of 2D model
                    # self.debug_viz( out_2D_2[i:i+b_2D], f"debug/output_2D_2_{i}")
                out_2D_3 = out_2D_3.reshape(b, w, self.out_c, d, h).permute(0, 2, 3, 4, 1)

                x_2D_1 = x.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
                t_2D_1 = torch.full((b*d,), int(timesteps.item()/self.time_step*self.ntime_steps_2D), device=x_2D_2.device, dtype=torch.long)
                out_2D_1 = torch.zeros(b*d, self.out_c, h, w, device=x_2D_1.device, dtype=x_2D_1.dtype)
                for i in range(0, b*d, b_2D):
                    # out_2D_1[i:i+b_2D] = self.model2D_1.denoise_inference(x_2D_1[i:i+b_2D,:self.out_c], t_2D_1[i:i+b_2D], y_cond=x_2D_1[i:i+b_2D,self.out_c:])
                    total_cond_array = []
                    for idx in range(i, i+b_2D):
                        data_array = []
                        for nearby_idx in range(-2,3):
                            data_array.append(self.slicing(x_2D_1, idx + nearby_idx)[self.out_c:self.out_c+num_modality])
                        data_array = torch.cat(data_array, dim = 0)
                        cond_array = torch.cat([data_array,x_2D_1[idx, self.out_c+num_modality:]], dim = 0)
                        total_cond_array.append(cond_array)
                    total_cond_array = torch.stack(total_cond_array)
                    # print(total_cond_array.shape) # 8, 9, 88, 64
                    # self.debug_viz(total_cond_array[0:1], f"debug/output_2D_1")
                    # raise
                    out_2D_1[i:i+b_2D] = self.model2D_1.denoise_inference(x_2D_1[i:i+b_2D,:self.out_c], t_2D_1[i:i+b_2D], y_cond=total_cond_array) # 8 1 88 64
                    # save the output of 2D model
                    # self.debug_viz( out_2D_2[i:i+b_2D], f"debug/output_2D_2_{i}")
                out_2D_1 = out_2D_1.reshape(b, d, self.out_c, h, w).permute(0, 2, 1, 3, 4)
                
            else: 
                # raise NotImplementedError("3D_feature is not supported for this basesline that use three 2.5D models")
                # Here the random crop size in all directionns have to be divisible by 2D batch size
                x_2D_2 = x.permute(0, 3, 1, 2, 4).reshape(-1, c, d, w)
                t_2D = torch.full((b*h,), int(timesteps.item()/self.time_step*self.ntime_steps_2D), device=x_2D_2.device, dtype=torch.long)
                
                # feature is calculated here
                input_c = x_2D_2.shape[1] - self.out_c
                img_idx = input_c - 3 - 1
                num_modality = img_idx # // 5
                
                out_2D_2 = torch.zeros(b*h, self.out_c, d, w, device=x_2D_2.device, dtype=x_2D_2.dtype)
                # hard coding num of channels for features :(
                out_feature_2_1 = torch.zeros(b*h, 512, 19, 24, device=x_2D_2.device, dtype=x_2D_2.dtype)
                out_feature_2_4 = torch.zeros(b*h, 256, 38, 48, device=x_2D_2.device, dtype=x_2D_2.dtype)
                out_feature_2_7 = torch.zeros(b*h, 128, 76, 96, device=x_2D_2.device, dtype=x_2D_2.dtype)
                out_feature_2_10 = torch.zeros(b*h, 64, 152, 192, device=x_2D_2.device, dtype=x_2D_2.dtype)
                # print("x_2D_2 shape", x_2D_2.shape): 64, 6, 88, 64]
                for i in range(0, b*h, b_2D):
                    total_cond_array = []
                    for idx in range(i, i+b_2D):
                        # needs to be changed when batch size is not 1
                        # x_2D_2[i:i+b_2D, self.out_c:]
                        data_array = []
                        for nearby_idx in range(-2,3):
                            data_array.append(self.slicing(x_2D_2, idx + nearby_idx)[self.out_c:self.out_c+num_modality])
                        data_array = torch.cat(data_array, dim = 0) # 5 x 1,88,64 -> 5,88,64 
                        cond_array = torch.cat([data_array,x_2D_2[idx,self.out_c+num_modality:]], dim = 0) # 5,88,64 + 4,88,64 -> 9,88,64
                        total_cond_array.append(cond_array)
                    total_cond_array = torch.stack(total_cond_array) # 8 x 9, 88, 64 -> 8, 9, 88, 64

                    # print(x_2D_2[i:i+b_2D,:self.out_c].shape, total_cond_array.shape) # 8, 1/9, 88, 64 / (152,192)
                    out_2D_2[i:i+b_2D], F = self.model2D_2.denoise_inference(x_2D_2[i:i+b_2D,:self.out_c], t_2D[i:i+b_2D], y_cond=total_cond_array,  feature = True) # 8 1 88 64
                    # for f_map in F:
                    #     print(f_map.shape)
                    # print("debugging for 2D f")
                    out_feature_2_1[i:i+b_2D] = F[1]
                    out_feature_2_4[i:i+b_2D] = F[4]
                    out_feature_2_7[i:i+b_2D] = F[7]
                    out_feature_2_10[i:i+b_2D] = F[10]                        
                    # raise
                    # save the output of 2D model
                    # self.debug_viz( out_2D_2[i:i+b_2D], f"debug/output_2D_2_{i}")
                out_2D_2 = out_2D_2.reshape(b, h, self.out_c, d, w).permute(0, 2, 3, 1, 4)
                out_feature_2_1 = out_feature_2_1.reshape(b, h, 512, 19, 24).permute(0, 2, 3, 1, 4)
                out_feature_2_4 = out_feature_2_4.reshape(b, h, 256, 38, 48).permute(0, 2, 3, 1, 4)
                out_feature_2_7 = out_feature_2_7.reshape(b, h, 128, 76, 96).permute(0, 2, 3, 1, 4)
                out_feature_2_10 = out_feature_2_10.reshape(b, h, 64, 152, 192).permute(0, 2, 3, 1, 4)

                
                x_2D_3 = x.permute(0, 4, 1, 2, 3).reshape(-1, c, d, h)
                out_2D_3 = torch.zeros(b*w, self.out_c, d, h, device=x_2D_3.device, dtype=x_2D_3.dtype)
                out_feature_3_1 = torch.zeros(b*w, 512, 19, 24, device=x_2D_2.device, dtype=x_2D_2.dtype)
                out_feature_3_4 = torch.zeros(b*w, 256, 38, 48, device=x_2D_2.device, dtype=x_2D_2.dtype)
                out_feature_3_7 = torch.zeros(b*w, 128, 76, 96, device=x_2D_2.device, dtype=x_2D_2.dtype)
                out_feature_3_10 = torch.zeros(b*w, 64, 152, 192, device=x_2D_2.device, dtype=x_2D_2.dtype)
                for i in range(0, b*h, b_2D):
                    # out_2D_3[i:i+b_2D] = self.model2D_3.denoise_inference(x_2D_3[i:i+b_2D,:self.out_c], t_2D[i:i+b_2D], y_cond=x_2D_3[i:i+b_2D,self.out_c:]) # 8 1 88 64
                    total_cond_array = []
                    for idx in range(i, i+b_2D):
                        data_array = []
                        for nearby_idx in range(-2,3):
                            data_array.append(self.slicing(x_2D_3, idx + nearby_idx)[self.out_c:self.out_c+num_modality])
                        data_array = torch.cat(data_array, dim = 0)
                        cond_array = torch.cat([data_array,x_2D_3[idx,self.out_c+num_modality:]], dim = 0)
                        total_cond_array.append(cond_array)
                    total_cond_array = torch.stack(total_cond_array)
                    # import pdb; pdb.set_trace()
                    # print(total_cond_array.shape) # 8, 9, 88, 64
                    # self.debug_viz(total_cond_array[0:1], f"debug/output_2D_3")
                    # raise
                    out_2D_3[i:i+b_2D], F = self.model2D_2.denoise_inference(x_2D_3[i:i+b_2D,:self.out_c], t_2D[i:i+b_2D], y_cond=total_cond_array, feature = True) # 8 1 88 64
                    out_feature_3_1[i:i+b_2D] = F[1]
                    out_feature_3_4[i:i+b_2D] = F[4]
                    out_feature_3_7[i:i+b_2D] = F[7]
                    out_feature_3_10[i:i+b_2D] = F[10]
                    # save the output of 2D model
                    # self.debug_viz( out_2D_2[i:i+b_2D], f"debug/output_2D_2_{i}")
                out_2D_3 = out_2D_3.reshape(b, w, self.out_c, d, h).permute(0, 2, 3, 4, 1)
                out_feature_3_1 = out_feature_3_1.reshape(b, w, 512, 19, 24).permute(0, 2, 3, 4, 1)
                out_feature_3_4 = out_feature_3_4.reshape(b, w, 256, 38, 48).permute(0, 2, 3, 4, 1)
                out_feature_3_7 = out_feature_3_7.reshape(b, w, 128, 76, 96).permute(0, 2, 3, 4, 1)
                out_feature_3_10 = out_feature_3_10.reshape(b, w, 64, 152, 192).permute(0, 2, 3, 4, 1)
                
                # out_feature_3_1 = self.feature_layer_3_1(out_feature_3_1)
                # out_feature_3_4 = self.feature_layer_3_4(out_feature_3_4)
                # out_feature_3_7 = self.feature_layer_3_7(out_feature_3_7)
                
                x_2D_1 = x.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
                t_2D_1 = torch.full((b*d,), int(timesteps.item()/self.time_step*self.ntime_steps_2D), device=x_2D_2.device, dtype=torch.long)
                out_2D_1 = torch.zeros(b*d, self.out_c, h, w, device=x_2D_1.device, dtype=x_2D_1.dtype)
                out_feature_1_1 = torch.zeros(b*d, 512, 24, 24, device=x_2D_2.device, dtype=x_2D_2.dtype)
                out_feature_1_4 = torch.zeros(b*d, 256, 48, 48, device=x_2D_2.device, dtype=x_2D_2.dtype)
                out_feature_1_7 = torch.zeros(b*d, 128, 96, 96, device=x_2D_2.device, dtype=x_2D_2.dtype)
                out_feature_1_10 = torch.zeros(b*d, 64, 192, 192, device=x_2D_2.device, dtype=x_2D_2.dtype)
                for i in range(0, b*d, b_2D):
                    # out_2D_1[i:i+b_2D] = self.model2D_1.denoise_inference(x_2D_1[i:i+b_2D,:self.out_c], t_2D_1[i:i+b_2D], y_cond=x_2D_1[i:i+b_2D,self.out_c:])
                    total_cond_array = []
                    for idx in range(i, i+b_2D):
                        data_array = []
                        for nearby_idx in range(-2,3):
                            data_array.append(self.slicing(x_2D_1, idx + nearby_idx)[self.out_c:self.out_c+num_modality])
                        data_array = torch.cat(data_array, dim = 0)
                        cond_array = torch.cat([data_array,x_2D_1[idx, self.out_c+num_modality:]], dim = 0)
                        total_cond_array.append(cond_array)
                    total_cond_array = torch.stack(total_cond_array)
                    # print(total_cond_array.shape) # 8, 9, 88, 64
                    # self.debug_viz(total_cond_array[0:1], f"debug/output_2D_1")
                    # raise
                    out_2D_1[i:i+b_2D], F = self.model2D_1.denoise_inference(x_2D_1[i:i+b_2D,:self.out_c], t_2D_1[i:i+b_2D], y_cond=total_cond_array, feature = True) # 8 1 88 64
                    out_feature_1_1[i:i+b_2D] = F[1]
                    out_feature_1_4[i:i+b_2D] = F[4]
                    out_feature_1_7[i:i+b_2D] = F[7]
                    out_feature_1_10[i:i+b_2D] = F[10]
                    # save the output of 2D model
                    # self.debug_viz( out_2D_2[i:i+b_2D], f"debug/output_2D_2_{i}")
                out_2D_1 = out_2D_1.reshape(b, d, self.out_c, h, w).permute(0, 2, 1, 3, 4)
                out_feature_1_1 = out_feature_1_1.reshape(b, d, 512, 24, 24).permute(0, 2, 1, 3, 4)
                out_feature_1_4 = out_feature_1_4.reshape(b, d, 256, 48, 48).permute(0, 2, 1, 3, 4)
                out_feature_1_7 = out_feature_1_7.reshape(b, d, 128, 96, 96).permute(0, 2, 1, 3, 4)
                out_feature_1_10 = out_feature_1_10.reshape(b, d, 64, 192, 192).permute(0, 2, 1, 3, 4)
                
                # out_feature_1_1 = self.feature_layer_1_1(out_feature_1_1)
                # out_feature_1_4 = self.feature_layer_1_4(out_feature_1_4)
                # out_feature_1_7 = self.feature_layer_1_7(out_feature_1_7)
                
                # print("debugging shape for 1")
                # print(out_feature_1_1.shape)
                # print(out_feature_1_4.shape)
                # print(out_feature_1_7.shape)
                # print(out_feature_1_10.shape)
                
               
                # print(out_feature_1.shape)
                # print(out_feature_4.shape)
                # print(out_feature_7.shape)
                # print(out_feature_10.shape)
                # raise
                # out_feature_1 = self.feature_extract_layer_1_1(out_feature_1_1) + self.feature_extract_layer_2_1(out_feature_2_1) + self.feature_extract_layer_3_1(out_feature_3_1)
                # out_feature_4 = self.feature_extract_layer_1_4(out_feature_1_4) + self.feature_extract_layer_2_4(out_feature_2_4) + self.feature_extract_layer_3_4(out_feature_3_4)
                # out_feature_7 = self.feature_extract_layer_1_7(out_feature_1_7) + self.feature_extract_layer_2_7(out_feature_2_7) + self.feature_extract_layer_3_7(out_feature_3_7)
                # torch.Size([1, 192, 19, 24, 24])
                # torch.Size([1, 128, 38, 48, 48])
                # torch.Size([1, 64, 76, 96, 96])
                # torch.Size([1, 64, 152, 192, 192])
                # raise
        if self.baseline == "2D":
            w2 = 0.0
            w3 = 1.0
            return w2 * out_2D_2 + w3 * out_2D_3
        elif self.baseline == "merge":
            w1 = 1.0 / 3
            w2 = 1.0 / 3
            w3 = 1.0 / 3
            return w1 * out_2D_1 + w2 * out_2D_2 + w3 * out_2D_3
        elif self.baseline == "TPDM":
            w2 = 0.1
            w3 = 0.666666
            # randomly pick one of the 2D model to use
            if np.random.rand() < w2:
                return out_2D_1
            elif np.random.rand() < w3:
                return out_2D_2
            else:
                return out_2D_3
            
        elif self.baseline == "3D":
            # inference for 3D model
            # print("x size:", x.shape)
            # print("2D out size 1:",out_2D_1.shape)
            # raise
            x_3D = torch.cat([x, out_2D_1.detach(), out_2D_2.detach(), out_2D_3.detach()], dim = 1)
            ensemble_w = self.model_3D(x_3D, timesteps)
            out_3D = (ensemble_w[:,0] + 0.3334) * out_2D_1 + (ensemble_w[:,1]+0.3333) * out_2D_2 +  (0.3333 - ensemble_w[:,0] - ensemble_w[:,1]) * out_2D_3 + ensemble_w[:,2]
            return out_3D
        elif self.baseline == "3D_only":
            # inference for 3D model only
            out_3D = self.model_3D(x, timesteps)
            return out_3D
        elif self.baseline == "3D_feature":
            # inference for 3D model
            # raise NotImplementedError("3D_feature is not supported for this basesline that use three 2.5D models")
            out_feature_1_1 = self.feature_layer_1_1(out_feature_1_1)
            out_feature_1_4 = self.feature_layer_1_4(out_feature_1_4)
            out_feature_1_7 = self.feature_layer_1_7(out_feature_1_7)
                
            out_feature_2_1 = self.feature_layer_2_1(out_feature_2_1)
            out_feature_2_4 = self.feature_layer_2_4(out_feature_2_4)
            out_feature_2_7 = self.feature_layer_2_7(out_feature_2_7)
            
            out_feature_3_1 = self.feature_layer_3_1(out_feature_3_1)
            out_feature_3_4 = self.feature_layer_3_4(out_feature_3_4)
            out_feature_3_7 = self.feature_layer_3_7(out_feature_3_7)
            
            out_feature_1 = out_feature_1_1 + out_feature_2_1 + out_feature_3_1
            out_feature_4 = out_feature_1_4 + out_feature_2_4 + out_feature_3_4
            out_feature_7 = out_feature_1_7 + out_feature_2_7 + out_feature_3_7
            out_feature_10 = self.feature_extract_layer_1_10(out_feature_1_10) + self.feature_extract_layer_2_10(out_feature_2_10) + self.feature_extract_layer_3_10(out_feature_3_10)
            out_features = [out_feature_1,out_feature_4,out_feature_7,out_feature_10]
            
            x_3D = torch.cat([x, out_2D_1.detach(), out_2D_2.detach(), out_2D_3.detach()], dim = 1)
            ensemble_w = self.model_3D(x_3D, timesteps, feature_2D = out_features)
            out_3D = (ensemble_w[:,0] + 0.3334) * out_2D_1 + (ensemble_w[:,1]+0.3333) * out_2D_2 +  (0.3333 - ensemble_w[:,0] - ensemble_w[:,1]) * out_2D_3 + self.weight_3D * ensemble_w[:,2]
            return out_3D
            # x_3D = torch.cat([x, out_2D_1.detach(), out_2D_2.detach(), out_2D_3.detach()], dim = 1)
            # feature_3D = torch.cat([feature_2D_2.detach(), feature_2D_3.detach()], dim = 1)
            # # x_3D_final = torch.cat([x_3D], dim = 1)
            
            # ensemble_w = self.model_3D(x_3D, timesteps, feature_2D = feature_3D)
            # out_3D = (ensemble_w[:,0] + 0.3334) * out_2D_1 + (ensemble_w[:,1]+0.3333) * out_2D_2 +  (0.3333 - ensemble_w[:,0] - ensemble_w[:,1]) * out_2D_3 + ensemble_w[:,2]
            # return out_3D
        else:
            raise ValueError("baseline must be 2D, 3D or merge")


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=3,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_2D_feature = False,
        feature_2D_channels = 128,
        small_model = False,
    ):
        super().__init__()
        
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        self.use_2D_feature = use_2D_feature
        if use_2D_feature:
            self.feature_2D_channels = [feature_2D_channels]

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float32
        self.use_fp16 = use_fp16
        print("use_fp16: ", self.use_fp16)
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1, padding_mode = 'replicate'))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for layers_idx in range(num_res_blocks):
                
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]

                # print("# in input_blocks", len(self.input_blocks))
                # print("init channels", ch)
                # print("additional channels", (self.feature_2D_channels[0] if (self.use_2D_feature and layers_idx == 1) else 0))
                # print("reason:", self.use_2D_feature, layers_idx)
                # print(layers)
                
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            # AttentionBlock(
            #     ch,
            #     use_checkpoint=use_checkpoint,
            #     num_heads=num_heads,
            #     num_head_channels=num_head_channels,
            #     use_new_attention_order=use_new_attention_order,
            # ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    dim_shift = True if (level == 4) else False
                    # print(level, dim_shift)
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            out_dim_shift_depth = dim_shift
                        )
                        if resblock_updown
                        # for the first up sample layer we need to have the same size as the last downsample layer
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch, out_dim_shift_depth=dim_shift)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1, padding_mode = 'replicate')),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, feature_2D = None, y=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        if feature_2D is not None:
            assert self.use_2D_feature

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled = self.use_fp16): 
            
            hs = []
            emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

            if self.num_classes is not None:
                assert y.shape == (x.shape[0],)
                emb = emb + self.label_emb(y)

            h = x #.type(self.dtype)
            # if not self.use_2D_feature:
            for i, module in enumerate(self.input_blocks):
                if self.use_2D_feature and (i == 2 or i == 5 or i == 8 or i == 10):
                    h = h + feature_2D.pop()
                    # h = th.cat([h, feature_2D], dim=1)
                # print("h shape:", h.shape)
                # print("==============================")
                # print(module)
                h = module(h, emb)
                hs.append(h)
            # raise
            
            h = self.middle_block(h, emb)
            for module in self.output_blocks:
                h = th.cat([h, hs.pop()], dim=1)
                h = module(h, emb)
            h = h #.type(x.dtype)
            return self.out(h)


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    in_channels=8,
    out_channels=4,
    use_2D_feature = False,
    feature_2D_channels = 128,
    small_model = False
):
    if channel_mult == "":
        if small_model:
            channel_mult = (1, 2, 2, 4)
            num_channels = num_channels // 2
        elif image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 192:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    if len(attention_resolutions) > 0:
        for res in attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=num_channels,
        out_channels=(1*out_channels if not learn_sigma else 2*out_channels),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
        use_2D_feature = use_2D_feature,
        feature_2D_channels = feature_2D_channels,
        small_model = small_model
    )
