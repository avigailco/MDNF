"""
The code is adapted from NFFB (https://github.com/ubc-vision/NFFB) (MIT)
"""
import math
import torch
import torch.nn as nn
from model.sine_layers import Sine, sine_init, first_layer_sine_init
from diffusion_net.layers import DiffusionNet


class MDNF_encoder(nn.Module):
    def __init__(self, encoding_config, network_config, n_input_dims,
                 diffusion_dropout=True, diffusion_outputs_at='vertices'):
        """
        :param encoding_config: DiffusionNet + Fourier parameters
        :param network_config: sine activated mlp network parameters
        :param n_input_dims: features input dimension to both first linear layer and DiffusionNet components
        :param diffusion_dropout: dropout parameter in DiffusionNet components
        :param diffusion_outputs_at: outputs_at parameter in DiffusionNet components
        """
        super().__init__()

        self.diffusion_outputs_at = diffusion_outputs_at
        sin_dims = network_config["dims"]
        if self.diffusion_outputs_at == 'vertices' or self.diffusion_outputs_at == 'faces':
            sin_dims = [n_input_dims] + sin_dims
        elif self.diffusion_outputs_at == 'edges':
            sin_dims = [n_input_dims * 2] + sin_dims
        self.num_sin_layers = len(sin_dims)

        ### The encoder part
        self.grid_level = int(self.num_sin_layers - 2)
        print(f"Number of resolution levels: {self.grid_level}")
        n_block = encoding_config["n_block"]  # number of diffusion blocks in each DiffusionNet component
        diffusion_hidden = encoding_config["diffusion_hidden"]  # size of diffusion hidden layers
        base_diffusion = encoding_config["base_diffusion"]  # initial diffusion time of lowest resolution component
        exp_diffusion = encoding_config["exp_diffusion"]  # diffusion time factor
        for layer in range(self.grid_level):
            init_t = base_diffusion * exp_diffusion ** layer
            setattr(self, "diffusion_layer" + str(layer), DiffusionNet(C_in=n_input_dims,
                                                                      C_width=diffusion_hidden,
                                                                      C_out=encoding_config["vertex_feat_dim"],
                                                                      N_block=n_block,
                                                                      outputs_at=diffusion_outputs_at,
                                                                      diffusion_method="spectral",
                                                                      dropout=diffusion_dropout,
                                                                      init_t=init_t,
                                                                      with_gradient_rotations=True))
        ### Create the ffn to map low-dim grid feats to map high-dim SIREN feats
        self.feat_dim = encoding_config["vertex_feat_dim"]  # fourier input features dim

        base_sigma = encoding_config["base_sigma"]
        exp_sigma = encoding_config["exp_sigma"]
        ffn_list = []
        for i in range(self.grid_level):
            ffn = torch.randn((self.feat_dim, sin_dims[2 + i]), requires_grad=True) * base_sigma * exp_sigma ** i
            ffn_list.append(ffn)
        self.ffn = nn.Parameter(torch.stack(ffn_list, dim=0))

        ### The low-frequency MLP part
        for layer in range(0, self.num_sin_layers - 1):
            setattr(self, "sin_lin" + str(layer), nn.Linear(sin_dims[layer], sin_dims[layer + 1]))
        self.sin_w0 = network_config["w0"]
        self.sin_activation = Sine(w0=self.sin_w0)
        self.init_siren()

        self.out_dim = sin_dims[-1] * self.grid_level

    def init_siren(self):
        for layer in range(0, self.num_sin_layers - 1):
            lin = getattr(self, "sin_lin" + str(layer))
            if layer == 0:
                first_layer_sine_init(lin)
            else:
                sine_init(lin, w0=self.sin_w0)

    def init_siren_out(self):
        for layer in range(0, self.grid_level):
            lin = getattr(self, "out_lin" + str(layer))
            sine_init(lin, w0=self.sin_w0_high)

    def forward(self, data_dict, edges_c=None):
        """
        :param data_dict: current example data dictionary
        :param edges_c (optional): edge pairs
        """
        min_val, max_val = torch.min(data_dict['feats']), torch.max(data_dict['feats'])
        x = 2 * (data_dict['feats'] - min_val) / (max_val - min_val) - 1

        # if self.diffusion_outputs_at == 'vertices' or self.diffusion_outputs_at == 'faces':
        #     x_in_first = x
        if self.diffusion_outputs_at == 'edges':
            x = torch.cat((x[edges_c[:, 0], :], x[edges_c[:, 1], :]), dim=1)

        device = x.device
        embedding_list = []
        k_eig = data_dict['evecs'].shape[-1]
        evecs_range = torch.linspace(0, k_eig, self.grid_level + 1).to(torch.int64)
        grid_x_list = []
        # diffusionNet components
        for layer in range(self.grid_level):
            diffusion_layer = getattr(self, "diffusion_layer" + str(layer))
            grid_x = diffusion_layer(data_dict['feats'], data_dict['mass'], L=data_dict['L'],
                                     evals=data_dict['evals'][..., evecs_range[layer]:evecs_range[layer + 1]],
                                     evecs=data_dict['evecs'][..., evecs_range[layer]:evecs_range[layer + 1]],
                                     gradX=data_dict['gradX'], gradY=data_dict['gradY'],
                                     edges=edges_c)
            grid_x_list.append(grid_x)

            # apply fourier
            grid_output = torch.matmul(grid_x, self.ffn[layer])
            grid_output = torch.sin(2 * math.pi * grid_output)
            embedding_list.append(grid_output)

        feat_list = []
        # ----------------------------------------------------------------------------------------------------
        # sin mlp
        for layer in range(0, self.num_sin_layers - 1):
            sin_lin = getattr(self, "sin_lin" + str(layer))
            x = sin_lin(x)
            x = self.sin_activation(x)
            if layer > 0:
                x = embedding_list[layer - 1] + x
                feat_list.append(x)

        x = feat_list
        return x


class MDNF(nn.Module):
    def __init__(self, config, n_input_dims=3, out_dims=1,
                 diffusion_dropout=True, diffusion_outputs_at='vertices'):
        """
        :param config: mdnf parameters
        :param n_input_dims: features input dimension to both first linear layer and DiffusionNet components
        :param out_dims: mdnf encoder output features dimension
        :param diffusion_dropout: dropout parameter in DiffusionNet components
        :param diffusion_outputs_at: outputs_at parameter in DiffusionNet components
        """
        super(MDNF, self).__init__()
        self.encoder = MDNF_encoder(n_input_dims=n_input_dims, encoding_config=config["encoding"],
                                    network_config=config["SIREN"], diffusion_dropout=diffusion_dropout,
                                    diffusion_outputs_at=diffusion_outputs_at)

        ### Initializing backbone part, to merge multi-scale features
        backbone_dims = config["Backbone"]["dims"]
        grid_feat_len = self.encoder.out_dim

        backbone_dims = [grid_feat_len] + backbone_dims + [out_dims]
        self.num_backbone_layers = len(backbone_dims)
        for layer in range(0, self.num_backbone_layers - 1):
            out_dim = backbone_dims[layer + 1]
            setattr(self, "backbone_lin" + str(layer), nn.Linear(backbone_dims[layer], out_dim))
        self.backbone_activation = config["Backbone"]["activation"]

    def forward(self, data):
        grid_x = self.encoder(data)
        out_feat = torch.cat(grid_x, dim=-1)  # concatenate encoder output features
        ### Backbone transformation
        for layer in range(0, self.num_backbone_layers - 1):
            backbone_lin = getattr(self, "backbone_lin" + str(layer))
            out_feat = backbone_lin(out_feat)
            if layer < self.num_backbone_layers - 2:
                out_feat = self.backbone_activation(out_feat)
        return out_feat

