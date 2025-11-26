import torch.nn as nn
from model.mdnf_model import MDNF


class UVLearningModel(nn.Module):
    """ Wrapper model for learning neural field representing UV function on a mesh """
    def __init__(self, config, n_input_dims=3, out_dims=2, diffusion_dropout=True, diffusion_outputs_at='vertices'):
        super(UVLearningModel, self).__init__()
        self.model = MDNF(config, n_input_dims=n_input_dims, out_dims=out_dims,
                          diffusion_dropout=diffusion_dropout,
                          diffusion_outputs_at=diffusion_outputs_at)

    def forward(self, diff_data):
        out_feat = self.model(diff_data)
        return out_feat
