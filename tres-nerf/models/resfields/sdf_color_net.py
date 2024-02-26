import torch
import torch.nn as nn
import numpy as np
import torch
import math
import torch.nn.functional as F
from instant_avatar.models.resfields.models.base import BaseModel
from instant_avatar.models.resfields import models
from instant_avatar.models.resfields.models.misc import get_rank
import instant_avatar.models.resfields.models.resfields as resfields

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()


    def create_embedding_fn(self):
        embed_fns = []
        self.input_dims = self.kwargs['input_dims']
        out_dim = 0
        self.include_input = self.kwargs['include_input']
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += self.input_dims

        max_freq = self.kwargs['max_freq_log2']
        self.num_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, self.num_freqs) * math.pi
        else:
            freq_bands = torch.linspace(2.**0.*math.pi, 2.**max_freq*math.pi, self.num_freqs)

        self.num_fns = len(self.kwargs['periodic_fns'])
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += self.input_dims

        self.embed_fns = embed_fns
        self.out_dim = out_dim


    # Anneal. Initial alpha value is 0, which means it does not use any PE (positional encoding)!
    def embed(self, inputs, alpha_ratio=0.):
        output = torch.cat([fn(inputs) for fn in self.embed_fns], -1)
        start = 0
        if self.include_input:
            start = 1
        for i in range(self.num_freqs):
            _dec = (1.-math.cos(math.pi*(max(min(alpha_ratio*self.num_freqs-i, 1.), 0.)))) * .5
            output[..., (self.num_fns*i+start)*self.input_dims:(self.num_fns*(i+1)+start)*self.input_dims] *= _dec
        return output


def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, alpha_ratio, eo=embedder_obj): return eo.embed(x, alpha_ratio)
    return embed, embedder_obj.out_dim

'''
'sdf_net': {'name': 'sdf_network', 'n_frames': 100, 'resfield_layers': [1, 2, 3, 4, 5, 6, 7], 
'composition_rank': 10, 'd_out': 129, 'd_in_1': 3, 'd_hidden': 128, 'n_layers': 8, 'skip_in': [4], 
'multires': 6, 'multires_topo': 0, 'bias': 0.5, 'scale': 1.0, 'geometric_init': True, 'weight_norm': False, 
'inside_outside': False, 'd_in_2': 0},
'''
import sys
from omegaconf import OmegaConf
config = OmegaConf.load('/public/home/gpu2002/dtl/InstantAvatar/confs/dataset/'+str(sys.argv[3]).split('=')[1]+'.yaml')
start = config.opt.train.start
end = config.opt.train.end+1
skip = config.opt.train.skip
# print(start, end, skip)
n_frame = len(range(1000)[start:end:skip])+1
print(n_frame)
# exit()
# n_frame = 0
# if str(sys.argv[3]).split('=')[1]=='female-4-casual':
#     n_frame=84+1
# elif str(sys.argv[3]).split('=')[1]=='male-4-casual':
#     n_frame=110+1
# elif str(sys.argv[3]).split('=')[1]=='male-3-casual':
#     n_frame=114+1
# elif str(sys.argv[3]).split('=')[1]=='female-3-casual':
#     n_frame=112+1
class SDFNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.rank = get_rank()

        self.n_frames = n_frame#self.config.n_frames
        self.capacity = self.n_frames
        self.d_out = 15#65#self.config.d_out
        self.d_in_1 = 3#self.config.d_in_1
        self.d_in_2 = 0#self.config.d_in_2
        self.d_hidden = 64#self.config.d_hidden
        self.n_layers = 8#self.config.n_layers
        self.skip_in = [4]#self.config.skip_in
        self.multires = 6#self.config.multires
        self.multires_topo = 0#self.config.multires_topo
        self.bias = 0.5#self.config.bias
        self.scale = 1.0
        self.geometric_init = True#self.config.geometric_init
        self.weight_norm = False#self.config.weight_norm
        self.inside_outside = False#self.config.inside_outside

        self.resfield_layers = [1,2,3,4,5,6,7]#self.config.get('resfield_layers', [])
        self.composition_rank = 10#self.config.get('composition_rank', 10)
        # create nets
        dims = [self.d_in_1 + self.d_in_2] + [self.d_hidden for _ in range(self.n_layers)] + [self.d_out]
        self.embed_fn_fine = None
        self.embed_amb_fn = None

        input_ch_1 = self.d_in_1
        input_ch_2 = self.d_in_2
        if self.multires > 0:
            embed_fn, input_ch_1 = get_embedder(self.multires, input_dims=self.d_in_1)
            self.embed_fn_fine = embed_fn
            dims[0] += (input_ch_1 - self.d_in_1)
        if self.multires_topo > 0:
            embed_amb_fn, input_ch_2 = get_embedder(self.multires_topo, input_dims=self.d_in_2)
            self.embed_amb_fn = embed_amb_fn
            dims[0] += (input_ch_2 - self.d_in_2)

        self.num_layers = len(dims)
        self.skip_in = self.skip_in
        self.scale = self.scale
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            _rank = self.composition_rank if l in self.resfield_layers else 0
            _capacity = self.capacity if l in self.resfield_layers else 0
            lin = resfields.Linear(dims[l], out_dim, rank=_rank, capacity=_capacity, mode='lookup')  # resfields模块调用

            if self.geometric_init:
                if l == self.num_layers - 2:
                    if not self.inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -self.bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, self.bias)
                elif self.multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, self.d_in_1:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :self.d_in_1], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    if self.multires > 0:
                        torch.nn.init.constant_(lin.weight[:, -(dims[0] - self.d_in_1):-input_ch_2], 0.0)
                    if self.multires_topo > 0:
                        torch.nn.init.constant_(lin.weight[:, -(input_ch_2 - self.d_in_2):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if self.weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
        self.activation = nn.Softplus(beta=100)

    def forward(self, input_pts, topo_coord=None, alpha_ratio=1.0, input_time=None, frame_id=None):
        """
            Args:
                input_pts: Tensor of shape (n_rays, n_samples, d_in_1)
                topo_coord: Optional tensor of shape (n_rays, n_samples, d_in_2)
                alpha_ratio (float): decay ratio of positional encoding
                input_time: Optional tensor of shape (n_rays, n_rays)
                # frame_id: Optional tensor of shape (n_rays)
        """
        # TIME = topo_coord
        inputs = input_pts * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs, alpha_ratio)
        if self.embed_amb_fn is not None:
            topo_coord = self.embed_amb_fn(topo_coord, alpha_ratio)
        if topo_coord is not None:
            inputs = torch.cat([inputs, topo_coord], dim=-1)
        x = inputs
        for l in range(0, self.num_layers - 1):
            if l in self.skip_in:
                x = torch.cat([x, inputs], -1) / np.sqrt(2)

            lin = getattr(self, "lin" + str(l))
            x = lin(x, input_time=input_time, frame_id=frame_id)

            if l < self.num_layers - 2:
                x = self.activation(x)
        sdf = (x[..., :1] / self.scale)
        out = torch.cat([sdf, x[..., 1:]], dim=-1)
        return out


    # Anneal
    def sdf(self, x, topo_coord, alpha_ratio, **kwargs):
        return self.forward(x, topo_coord, alpha_ratio, **kwargs)[..., :1]

    def sdf_hidden_appearance(self, x, topo_coord, alpha_ratio, **kwargs):
        return self.forward(x, topo_coord, alpha_ratio, **kwargs)

    def gradient(self, x, topo_coord, alpha_ratio, **kwargs):
        x.requires_grad_(True)
        y = self.sdf(x, topo_coord, alpha_ratio, **kwargs)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients

'''
'color_net': {'name': 'color_network', 'd_hidden': 128, 'n_layers': 1, 'mode': {'feature': 128}}, 

'deviation_net': {'name': 'laplace_density', 'beta_min': 0.0001, 'beta': 0.1}, 
'metadata': {'scene_aabb': [-0.4352, -0.9157, -0.4957, 0.5201, 0.9316, 0.703], 'n_frames': 100}}

{'name': 'DySDF', 'background': 'black', 'ambient_dim': 0, 'deform_dim': 0, 
'alpha_ratio': {'strategy': 'interpolated', 'max_steps': 50000}, 
'isosurface': {'resolution': 512}, 
'sampling': {'strategy': 'balanced', 'train_num_rays': 1100, 'n_samples': 256, 
'n_importance': 0, 'randomized': True, 'ray_chunk': 1100}, 
'deform_net': None, 
'''

class ColorNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.rank = get_rank()

        supported_modes = ['feature', 'point', 'ambient', 'view', 'normal']

        # get params
        self.mode = {'feature':64}#self.config.mode # dictionary: encoding_type,feature_dim
        assert len(self.mode) != 0, 'No input features specified'
        d_hidden = 64#self.config.d_hidden
        n_layers = 1#self.config.n_layers

        weight_norm = False#self.config.get('weight_norm', False)
        d_out = 3#self.config.get('d_out', 3)
        multires_view = 0#self.config.get('multires_view', 0)

        # create encodings
        f_in = 0
        for encoding, encoding_dim in self.mode.items():
            assert encoding in supported_modes, f'Encoding {encoding} not supported'
            f_in += encoding_dim

        self.embedview_fn = None
        if multires_view > 0 and 'view' in self.mode:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            f_in += (input_ch - self.config.dim_view)

        # create network
        dims = [f_in] + [d_hidden for _ in range(n_layers)] + [d_out]
        self.num_layers = len(dims)
        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
        self.relu = nn.ReLU()

    def forward(self, feature=None, point=None, ambient_code=None, view_dir=None, normal=None, alpha_ratio=1.0):
        # concatentate all inputs
        input_encoding = []
        if 'feature' in self.mode:
            input_encoding.append(feature)
        if 'point' in self.mode:
            input_encoding.append(point)
        if 'ambient' in self.mode:
            input_encoding.append(ambient_code)
        if 'view' in self.mode:
            if self.embedview_fn is not None:
                view_dirs = self.embedview_fn(view_dirs, alpha_ratio) # Anneal
            input_encoding.append(view_dir)
        if 'normal' in self.mode:
            input_encoding.append(normal)
        x = torch.cat(input_encoding, dim=-1)

        # forward through network
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.relu(x)
        x = torch.sigmoid(x)
        return x

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Attention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_in = nn.Linear(input_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        # 输入inputs的形状：(batch_size, seq_len, input_size)
        inputs = inputs.unsqueeze(1)
        batch_size, seq_len, _ = inputs.size()

        # 将输入通过线性层变换：(batch_size, seq_len, hidden_size)
        hidden = torch.tanh(self.linear_in(inputs))

        # 计算注意力权重
        # 将hidden变形为(batch_size * seq_len, hidden_size)
        hidden = hidden.view(batch_size * seq_len, -1)
        # 通过线性层变换得到注意力分数：(batch_size * seq_len, 1)
        attn_scores = self.linear_out(hidden)
        # 将注意力分数变形为(batch_size, seq_len, 1)
        attn_scores = attn_scores.view(batch_size, seq_len, 1)
        # 使用softmax函数计算注意力权重：(batch_size, seq_len, 1)
        attn_weights = F.softmax(attn_scores, dim=1)

        # 将注意力权重应用于输入特征
        # 将输入特征与注意力权重相乘：(batch_size, seq_len, input_size)
        # print(inputs.shape, attn_weights.shape, 111111111111)
        weighted_inputs = inputs * attn_weights
        # 对加权输入特征进行求和，得到注意力加权后的表示：(batch_size, input_size)
        attn_output = torch.sum(weighted_inputs, dim=1)

        return attn_output

'''
'hyper_net': {'name': 'hyper_time_network', 'd_in': 0, 'd_out': 0, 'multires_out': 0},
'''
class HyperNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.rank = get_rank()

        self.d_in = 64
        self.multires_out = 0
        self.out_dim = 64
        if self.multires_out > 0:
            self.embed_fn, self.out_dim = get_embedder(self.multires_out, input_dims=self.config.d_out)
        else:
            self.embed_fn = None

    def _forward(self, deformation_code, input_pts, input_time, alpha_ratio):
        raise NotImplementedError

    def forward(self, deformation_code, input_pts, input_time, alpha_ratio):
        if self.d_in == 0:
            return None
        out = self._forward(deformation_code, input_pts, input_time, alpha_ratio)
        if self.embed_fn is not None:
            out = self.embed_fn(out, alpha_ratio)
        return out

class HyperCondNetwork(HyperNetwork):

    def _forward(self, deformation_code, input_pts, input_time, alpha_ratio):
        return deformation_code


'''
    d_time: 1
    d_in: 3
    multires: 0
    depth: 8
    width: 128
    skips: [4,]
'''
class DeformDNeRF(nn.Module):
    """ Model deformation like in DNeRF 
    """

    def __init__(self):
        super().__init__()
        self.rank = get_rank()

        d_time = 1
        d_in = 3
        depth = 8
        width = 128
        skips = [4,]
        multires = 0
        if multires > 0:
            self.embed_fn, d_in = get_embedder(multires, input_dims=d_in)
        else:
            self.embed_fn = lambda x, y: x

        self.time_mlp = MLP(
            in_ch=d_in+d_time,
            depth=depth,
            width=width,
            skips=skips,
            out_ch=d_in,
            act=torch.relu
        )

    def forward(self, deformation_code, input_pts, alpha_ratio, time_step):
        points = self.embed_fn(input_pts, alpha_ratio)
        input_x = torch.cat((points, time_step), dim=-1)
        dx = self.time_mlp(input_x)
        return input_pts + dx
    
class MLP(torch.nn.Module):
    def __init__(self,
            in_ch: int,
            depth: int, 
            width: int, 
            out_ch: int = 0,
            skips: tuple = tuple(), 
            act=torch.relu, 
            out_act=lambda x: x
        ) -> None:
        super().__init__()
        self.act = act
        self.out_act = out_act
        self.skips = skips

        layers = [torch.nn.Linear(in_ch, width)]
        for i in range(depth - 1):
            in_channels = width
            if i in skips:
                in_channels += in_ch

            layers.append(torch.nn.Linear(in_channels, width))
        self.net = torch.nn.ModuleList(layers)
        if out_ch > 0:
            self.net_out = torch.nn.Linear(width, out_ch)
        else:
            self.net_out = lambda x: x

    def forward(self, input_x):
        h = input_x
        for i, l in enumerate(self.net):
            h = self.net[i](h)
            h = self.act(h)
            if i in self.skips:
                h = torch.cat([input_x, h], -1)

        return self.out_act(self.net_out(h))
