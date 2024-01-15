import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from seqtr.models import VIS_ENCODERS
from seqtr.utils import get_root_logger, is_main
from seqtr.models.utils import freeze_params, parse_yolo_weights
import torch
import torch.nn as nn
import torchvision
 
 
class NonLocalBlock(nn.Module):
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)
 
    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b,self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out

class Vis_SA(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
 
    def forward(self,x):
        """
        inputs :
            x : input feature maps (B x C x W x H)
        returns :
            out : self attention value + input feature
            attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0,2,1) # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height) # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height) # B X C X N
 
        out = torch.bmm(proj_value,attention.permute(0, 2, 1) )
        out = out.view(m_batchsize, C, width, height)
 
        out = self.gamma * out + x
 
        return out, attention

def darknet_conv(in_chs,
                 out_chs,
                 kernel_sizes,
                 strides,
                 norm_cfg=dict(type="BN2d"),
                 act_cfg=dict(type="LeakyReLU", negative_slope=0.1)):
    convs = []
    for i, (in_ch, out_ch, kernel_size, stride) in enumerate(zip(in_chs, out_chs, kernel_sizes, strides)):
        convs.append(ConvModule(in_ch,
                                out_ch,
                                kernel_size,
                                stride=stride,
                                padding=kernel_size // 2,
                                norm_cfg=norm_cfg,
                                act_cfg=act_cfg))

    return convs


class DarknetBlock(nn.Module):
    def __init__(self,
                 ch,
                 num_block=1,
                 shortcut=True):
        super(DarknetBlock, self).__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList([
            nn.ModuleList([*darknet_conv((ch, ch // 2),
                          (ch // 2, ch), (1, 3), (1, 1))])
            for _ in range(num_block)
        ])

    def forward(self, x):
        for module in self.module_list:
            residual = x
            for conv in module:
                x = conv(x)
            if self.shortcut:
                x = x + residual
        return x


@VIS_ENCODERS.register_module()
class DarkNet53(nn.Module):
    def __init__(self,
                 freeze_layer=2,
                 pretrained='./data/weights/yolov3.weights',
                 out_layer=(6, 8, 13)):
        super(DarkNet53, self).__init__()
        self.fp16_enabled = False
        assert isinstance(out_layer, tuple)
        self.out_layer = out_layer

        self.darknet = nn.ModuleList([
            *darknet_conv((3, 32), (32, 64), (3, 3), (1, 2)),
            DarknetBlock(64),
            *darknet_conv((64, ), (128, ), (3, ), (2, )),
            DarknetBlock(128, num_block=2),
            *darknet_conv((128, ), (256, ), (3, ), (2, )),
            DarknetBlock(256, num_block=8),
            *darknet_conv((256, ), (512, ), (3, ), (2, )),
            DarknetBlock(512, num_block=8),
            *darknet_conv((512, ), (1024, ), (3, ), (2, )),
            DarknetBlock(1024, num_block=4),
            DarknetBlock(1024, num_block=2, shortcut=False),
            *darknet_conv((1024, 512), (512, 1024), (1, 3), (1, 1))
        ])
        # self.vis_global = NonLocalBlock(256)
        if pretrained is not None:
            parse_yolo_weights(self, pretrained, len(self.darknet))
            if is_main():
                logger = get_root_logger()
                logger.info(
                    f"load pretrained visual backbone from {pretrained}")

        self.do_train = False
        if freeze_layer is not None:
            freeze_params(self.darknet[:-freeze_layer])
        else:
            self.do_train = True

    @force_fp32(apply_to=('img', ))
    def forward(self, img):
        x = []
        for i, mod in enumerate(self.darknet):
            img = mod(img)
            if i in self.out_layer:
                x.append(img)

        if len(self.out_layer) == 1:
            return x[0]
        else:
            # x[0]= self.vis_global(x[0])
            # print(f"forward darknet53 x return layer(6): {x[0].shape}") # shape [batchsize, 256, 80/72/68/64, 80])
            # print(f"forward darknet53 x return layer(8): {x[1].shape}") # shape [batchsize, 512, 40/36/34/32, 40])
            # print(f"forward darknet53 x return layer(13): {x[2].shape}")# shape [batchsize, 1024,20/18/17/16, 20])
            return x
