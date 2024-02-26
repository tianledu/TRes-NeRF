import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
import tinycudann as tcnn
import torch.nn.functional as F
from instant_avatar.models.resfields.CVAE import CVAE

EPS = 1e-3


class TruncExp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x.clamp(max=15, min=-15))

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(max=15, min=-15))

class NeRFNGPNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        self.encoder = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=16,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.5,
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )

        self.color_net = tcnn.Network(
            n_input_dims=15,#+65,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        )
        # self.sigma_activ = lambda x: torch.relu(x).float()
        # self.sigma_activ = TruncExp.apply
        self.register_buffer("center", torch.FloatTensor(opt.center))
        self.register_buffer("scale", torch.FloatTensor(opt.scale))
        self.opt = opt

        self.cvae = CVAE(feature_size=1, y_input_size=3, latent_size=32)

        # ==========
        # from instant_avatar.models.resfields import sdf_color_net
        # self.sdf_net = sdf_color_net.SDFNetwork()#.cuda()
        # self.color_res_net = sdf_color_net.ColorNetwork()#.cuda()
        # self.attention_model = sdf_color_net.Attention(15+16, 96)
        # print(self.sdf_net)
        # print(self.color_net)
        # ==========

    def initialize(self, bbox):
        if hasattr(self, "bbox"):
            return
        c = (bbox[0] + bbox[1]) / 2
        s = (bbox[1] - bbox[0])
        self.center = c
        self.scale = s
        self.bbox = bbox

    def forward(self, x, d, batch=None):
        embed, res, mu, log_std = 0, 0, 0, 0
        if batch != None:
            cond = batch['frame_time'][0].to(torch.float32)
            # # print(x.shape, cond.shape)
            cond = torch.repeat_interleave(cond, x.shape[0]).reshape(x.shape[0], -1)
            embed, res, mu, log_std = self.cvae(cond, x.reshape(-1, 3))
            # print(x.shape, res.shape)
            x = x + res.reshape(-1, 3)  # * 0.01

        # normalize pts and view_dir to [0, 1]
        x = (x - self.center) / self.scale + 0.5
        # assert x.min() >= -EPS and x.max() < 1 + EPS
        x = x.clamp(min=0, max=1)

        # ================================================ 
        # 尝试改为sdf_network和color_network
        # 计算sdf
        # rgb rays_o rays_d betas global_orient body_pose transl alpha bg_color idx frame_time near far
        # if batch == None:
        #     frame_id = 0
        #     view_dir=torch.zeros(size=(x.shape[0], 1))
        #     pts = torch.zeros_like(x)
        #     input_time = None
        # else:
        #     frame_id = batch['idx'][0]#.to(torch.float32)
        #     view_dir=batch['rays_d']
        #     pts = batch['pts']
        #     input_time = batch['frame_time']
        # sdf_nn_output = self.sdf_net(x, None, alpha_ratio=1.0, input_time=input_time, frame_id=frame_id)
        # sdf, feature_vector = sdf_nn_output[..., :1], sdf_nn_output[..., 1:] # (n_rays, n_samples, 1), (n_rays, n_samples, F)
        # # x = x + torch.tanh(sdf_nn_output[..., :3].reshape(-1, 3))*0.1
        # self.estimate_normals = False
        # if self.estimate_normals and pts!=None:
        #     pts.requires_grad_(True)
        #     gradients_o =  torch.autograd.grad(outputs=sdf, inputs=pts, grad_outputs=torch.ones_like(sdf, requires_grad=False, device=sdf.device), create_graph=True, retain_graph=True, only_inputs=True)[0]
        # else:
        #     gradients_o = None
        # # print(self.sdf_net)
        # # print(self.color_net)
        # # 计算采样点颜色【使用sdf网络替代了常规nerf模型的体密度】，考虑对feature_vector使用注意力机制
        # color_ref = self.color_res_net(feature=feature_vector, point=x, ambient_code=None, view_dir=view_dir, normal=gradients_o, alpha_ratio=1.0) # n_rays, n_samples, 3
        # sigma=torch.relu(sdf.reshape(-1))
        # # volume rendering 体渲染
        # weights = self.get_weight(sdf, dists) # n_rays, n_samples, 1
                
        # comp_rgb = (color * weights).sum(dim=1) # n_rays, 3
        # opacity = weights.sum(dim=1) # n_rays, 1
        # depth = (weights*mid_z_vals.unsqueeze(-1)).sum(dim=1) # n_rays, 1
        # exit()
        # ================================================ 

        x = self.encoder(x)
        # x = self.density_net(x)
        sigma = x[..., 0]
        
        # exit()torch.cat([x[..., 1:], self.attention_model(color_ref)], dim=-1)self.attention_model(
        # print(torch.cat([self.attention_model(x[..., 1:]), color_ref], dim=-1))torch.cat([x[..., 1:], torch.relu(sdf_nn_output)], dim=-1)
        color = self.color_net(x[..., 1:]).float()
        # color += color_ref*0.01  # 考虑均值
        # sigma = self.sigma_activ(sigma)
        result = {
                # 'ref':color_ref, 
                'embed':embed, 
                'res':res, 
                'mu':mu, 
                'log_std':log_std
        }
        return color, sigma.float(), result
