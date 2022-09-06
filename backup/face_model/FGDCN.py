import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d


def flow_warp(Fref,flow_P,interpolation='bilinear',padding_mode='zeros',align_corners=False):
    """Warp an image or a feature map with flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.

    Returns:
        Tensor: Warped image or feature map.
    """
    if Fref.size()[-2:] != flow_P.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({Fref.size()[-2:]}) and 'f'flow ({flow_P.size()[1:3]}) are not the same.')

    _, _, h, w = Fref.size()

    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * flow_P[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * flow_P[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(
        Fref,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)
    return output


class FlowGuidedDeformableAlignment(ModulatedDeformConv2d):
    """FlowGuidedDeformableAlignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.

    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(FlowGuidedDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 18 * self.deform_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, Ftex, Fsisr, flow_P):
        extra_feat = torch.cat([Ftex, Fsisr], dim=1)
        out = self.conv_offset(extra_feat)
        o1, mask = torch.chunk(out, 2, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(o1)
        offset = offset + flow_P.flip(1).repeat(1,offset.size(1)//2,1,1)
        
        # mask
        mask = torch.sigmoid(mask)

        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)

if __name__ == "__main__":
    """
    #test FGDCN
    device = "cuda"
    mid_channels = 64

    DCN_mod = FlowGuidedDeformableAlignment(
        in_channels=mid_channels*2,
        out_channels=mid_channels,
        kernel_size=3,
        padding=1,
        deform_groups=16,
        max_residue_magnitude=10
        )
    DCN_mod = DCN_mod.to(device)

    feat_x = torch.randn(1,mid_channels*2,256,256).to(device)
    F_tex = torch.randn(1,mid_channels,256,256).to(device)
    F_sisr = torch.randn(1,mid_channels,256,256).to(device)
    flow_p = torch.tanh(torch.randn(1,2,256,256)).to(device)

    y = DCN_mod(feat_x,F_tex,F_sisr,flow_p)
    print(y.shape)
    """

    #test flow warp
    F_ref = torch.randn(1,64,256,256)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, 256, dtype=F_ref.dtype),
        torch.arange(0, 256, dtype=F_ref.dtype))
    Flow_p = torch.stack((grid_x, grid_y), 2)  # h, w, 2
    Flow_p = Flow_p.unsqueeze(0)

    y = flow_warp(F_ref, Flow_p)
    print(y.shape)
