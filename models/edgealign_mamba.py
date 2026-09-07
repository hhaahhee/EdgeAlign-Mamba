import os

from .vmamba import VSSM
import torch
from torch import nn


class EdgeAlignMamba(nn.Module):
    """EdgeAlign-Mamba segmentation network built on the VM-UNet backbone."""
    def __init__(self, 
                 input_channels=3, 
                 num_classes=1,
                 depths=[2, 2, 2, 2], 
                 depths_decoder=[2, 2, 2, 1],
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                 use_mbam: bool = True,
                 mbam_stages=(0, 1, 2),
                 mbam_on_skip: bool = False,
                 mbam_branch_ratio: int = 4,
                 mbam_dilation: int = 2,
                 mbam_ca_ratio: int = 8,
                 mbam_branch_mask=(1, 1, 1),
                 mbam_use_boundary_branch: bool = True,
                 mbam_use_scale_selection: bool = True,
                 mbam_use_channel_attention: bool = True,
                 mbam_use_edge_gate: bool = True,
                 mbam_use_res_scale: bool = True,
                 use_dual_skip: bool = True,
                 dual_skip_stages=(1, 2, 3),
                 mamba_skip_gate_ratio: int = 8,
                ):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes

        # Keep the historical attribute name so existing checkpoints remain loadable.
        self.vmunet = VSSM(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate,
                                    use_mbam=use_mbam,
                                    mbam_stages=mbam_stages,
                                    mbam_on_skip=mbam_on_skip,
                                    mbam_branch_ratio=mbam_branch_ratio,
                                    mbam_dilation=mbam_dilation,
                                    mbam_ca_ratio=mbam_ca_ratio,
                                    mbam_branch_mask=mbam_branch_mask,
                                    mbam_use_boundary_branch=mbam_use_boundary_branch,
                                    mbam_use_scale_selection=mbam_use_scale_selection,
                                    mbam_use_channel_attention=mbam_use_channel_attention,
                                    mbam_use_edge_gate=mbam_use_edge_gate,
                                    mbam_use_res_scale=mbam_use_res_scale,
                                    use_dual_skip=use_dual_skip,
                                    dual_skip_stages=dual_skip_stages,
                                    mamba_skip_gate_ratio=mamba_skip_gate_ratio,
                        )
    
    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.vmunet(x)
        if self.num_classes == 1:
            return torch.sigmoid(logits)
        return logits
    
    def load_from(self):
        if self.load_ckpt_path is not None:
            if not os.path.isfile(self.load_ckpt_path):
                raise FileNotFoundError(
                    f"Pre-trained VMamba checkpoint not found: {self.load_ckpt_path}. "
                    "Download vmamba_small_e238_ema.pth from the official "
                    "VM-UNet resources and place it at the configured path. "
                    "See README.md for details."
                )
            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path, map_location='cpu')
            pretrained_dict = modelCheckpoint['model']
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
            self.vmunet.load_state_dict(model_dict)

            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("encoder loaded finished!")

            model_dict = self.vmunet.state_dict()
            pretrained_odict = modelCheckpoint['model']
            pretrained_dict = {}
            for k, v in pretrained_odict.items():
                if 'layers.0' in k: 
                    new_k = k.replace('layers.0', 'layers_up.3')
                    pretrained_dict[new_k] = v
                elif 'layers.1' in k: 
                    new_k = k.replace('layers.1', 'layers_up.2')
                    pretrained_dict[new_k] = v
                elif 'layers.2' in k: 
                    new_k = k.replace('layers.2', 'layers_up.1')
                    pretrained_dict[new_k] = v
                elif 'layers.3' in k: 
                    new_k = k.replace('layers.3', 'layers_up.0')
                    pretrained_dict[new_k] = v
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
            self.vmunet.load_state_dict(model_dict)
            
            # 找到没有加载的键(keys)
            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("decoder loaded finished!")


# Compatibility alias for earlier experiment scripts.
VMUNet = EdgeAlignMamba
