from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

import sys, os
root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from acr.config import args
from acr.result_parser import ResultParser
from acr.utils import BHWC_to_BCHW
if args().model_precision=='fp16':
    from torch.cuda.amp import autocast

BN_MOMENTUM = 0.1

class ACR(nn.Module):
    def __init__(self, **kwargs):
        super(ACR, self).__init__()
        print('Using ACR v1')
        self.backbone = HigherResolutionNet() # backbone
        self._result_parser = ResultParser() # manolayer and post-processing
        self._build_head()

    @torch.no_grad()
    def forward(self, meta_data, **cfg):
        if args().model_precision=='fp16':
            with autocast():
                x = self.backbone(meta_data['image'].contiguous().cuda())
                outputs = self.head_forward(x)
                outputs, meta_data = self._result_parser.parse(outputs, meta_data, cfg)
        else:
            x = self.backbone(meta_data['image'].contiguous().cuda())
            outputs = self.head_forward(x)
            outputs, meta_data = self._result_parser.parse(outputs, meta_data, cfg)

        outputs['meta_data'] = meta_data
        return outputs

    @torch.no_grad()
    def head_forward(self,x, gt_segm=None):

        pred_segm = self.backbone.hand_segm(x)

        # cat coords
        x = torch.cat((x, self.coordmaps.to(x.device).repeat(x.shape[0],1,1,1)), 1)

        l_params_maps, r_params_maps, l_center_maps, r_center_maps, l_prior_maps, r_prior_maps = self.global_forward(x)
        l_params_maps, r_params_maps, segms = self.part_forward(x, gt_segm, pred_segm, l_params_maps, r_params_maps)
        output = {'l_params_maps':l_params_maps.float(),
                'r_params_maps':r_params_maps.float(),
                'l_center_map':l_center_maps.float(),
                'r_center_map':r_center_maps.float(),
                'l_prior_maps':l_prior_maps.float() if args().inter_prior else None, 
                'r_prior_maps':r_prior_maps.float() if args().inter_prior else None,
                'segms':segms.float() if 'pred' in args().attention_mode else None
                }

        return output

    @torch.no_grad()
    def global_forward(self, x):

        # split into left and right
        l_params_maps = self.l_final_layers[1](x)
        l_center_maps = self.l_final_layers[2](x)
        if args().inter_prior:
            l_prior_maps = self.l_final_layers[4](x)
        else:
            l_prior_maps = None

        r_params_maps = self.r_final_layers[1](x)
        r_center_maps = self.r_final_layers[2](x)
        if args().inter_prior:
            r_prior_maps = self.r_final_layers[4](x)
        else:
            r_prior_maps = None

        if args().merge_mano_camera_head:
            print('Merging head not applicable')
            raise NotImplementedError
            l_cam_maps, r_params_maps = l_params_maps[:,:3], l_params_maps[:,3:]
            r_cam_maps, r_params_maps = r_params_maps[:,:3], r_params_maps[:,3:]
        else:
            l_cam_maps = self.l_final_layers[3](x)
            r_cam_maps = self.r_final_layers[3](x)

        # to make sure that scale is always a positive value
        l_cam_maps[:, 0] = torch.pow(1.1,l_cam_maps[:, 0])
        r_cam_maps[:, 0] = torch.pow(1.1,r_cam_maps[:, 0])

        l_params_maps = torch.cat([l_cam_maps, l_params_maps], 1)
        r_params_maps = torch.cat([r_cam_maps, r_params_maps], 1)

        return l_params_maps, r_params_maps, l_center_maps, r_center_maps, l_prior_maps, r_prior_maps

    def Hadamard_product(self, features, heatmaps):
        batch_size, num_joints, height, width = heatmaps.shape

        normalized_heatmap = F.softmax(heatmaps.reshape(batch_size, num_joints, -1), dim=-1)

        features = features.reshape(batch_size, -1, height*width)

        attended_features = torch.matmul(normalized_heatmap, features.transpose(2,1))
        attended_features = attended_features.transpose(2,1)

        return attended_features

    @torch.no_grad()
    def part_forward(self, x, gt_segm, pred_segm, l_params_maps, r_params_maps):
        ####################################################
        ################## contact offsets #################
        ####################################################
        # pose: x -> deconv -> attention -> regress
        # shape: x -> deconv -> conv to 64, here use the same deconv as pose, deconv_contact_features
        # segm: deconv -> regress -> softtmax()
        BS = len(x)

        ############## 2D PART BRANCH FEATURES ##############
        part_attention = F.interpolate(pred_segm.clone().float(), scale_factor=(1/2, 1/2), mode='nearest') # because pred_segm is [256, 256], part-attention need to be 128
        logits = pred_segm
        part_attention = part_attention[:,1:,:,:]

        ############## 3D SMPL BRANCH FEATURES ##############
        deconv_contact_features = self.contact_layers[1](x) # from Bx (128+2)x128x128 -> Bx 256x128x128
        deconv_shape_features = self.cam_shape_layers[1](deconv_contact_features) # from Bx 256x128x128 -> Bx 64x128x128, (smpl_final_layer)

        ############## SAMPLE LOCAL FEATURES ##############
        weighted_contact_features = self.Hadamard_product(deconv_contact_features, part_attention) # Hadamard product: (Bx256x128x128) * σ(BxJx128x128) -> Bx 256xJ
        weighted_shape_features = self.Hadamard_product(deconv_shape_features, part_attention) # Hadamard product: (Bx256xHxW) * σ(BxJxHxW) -> Bx 64xJ
        weighted_contact_features = weighted_contact_features.unsqueeze(-1)  # Bx 256xJ -> Bx 256xJx1

        ############ SORT LEFT AND RIGHT ORDER #############
        # pose feature
        l_weighted_contact_features, r_weighted_contact_features = weighted_contact_features[:, :, 16:, :], weighted_contact_features[:, :, :16, :] # Bx 256x16x1

        # shape feature
        l_weighted_shape_features, r_weighted_shape_features = weighted_shape_features[:, :, 16:], weighted_shape_features[:, :, :16]
        l_weighted_shape_features = torch.flatten(l_weighted_shape_features, start_dim=1)  # Bx (64*16)
        r_weighted_shape_features = torch.flatten(r_weighted_shape_features, start_dim=1)  # Bx (64*16)

        ############## GET FINAL PREDICTIONS ##############
        # left
        l_contact_offsets = self.contact_layers[2](l_weighted_contact_features).squeeze(-1).transpose(2, 1).reshape(BS, 96) # Bx 256xJx1 conv-> [Bx(96)xJx1 or  B x 6 x J x1] -> [Bx1x96 or BxJx6] -reshape>  [Bx96, Bx96]
        l_shape_offsets = self.cam_shape_layers[2](l_weighted_shape_features) # Bx (64*J) -> Bx(10)

        # right
        r_contact_offsets = self.contact_layers[3](r_weighted_contact_features).squeeze(-1).transpose(2, 1).reshape(BS, 96) # Bx 256xJx1 -> [Bx(96)xJx1 or  B x 6 x J x1] -> [Bx1x96 or BxJx6] -reshape> [Bx96, Bx96]
        r_shape_offsets = self.cam_shape_layers[3](r_weighted_shape_features) # Bx (64*J) -> Bx(10)

        # replicate original camera for concat
        l_pare_features = torch.cat((l_contact_offsets, l_shape_offsets), dim=1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 64, 64) # Bx109x1x1 -> Bx109x64x64
        r_pare_features = torch.cat((r_contact_offsets, r_shape_offsets), dim=1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 64, 64) # Bx109x1x1 -> Bx109x64x64
        l_params_pare = torch.cat((l_params_maps[:, :3].clone(), l_pare_features), dim=1)
        r_params_pare = torch.cat((r_params_maps[:, :3].clone(), r_pare_features), dim=1)

        l_params_maps = self.contact_layers[4](torch.cat((l_params_maps, l_params_pare), dim=1))
        r_params_maps = self.contact_layers[5](torch.cat((r_params_maps, r_params_pare), dim=1))

        return l_params_maps, r_params_maps, logits

    def _build_head(self):
        self.outmap_size = args().centermap_size
        params_num, cam_dim = self._result_parser.params_num, 3
        self.head_cfg = {'NUM_HEADS': 1, 'NUM_CHANNELS': 64, 'NUM_BASIC_BLOCKS': args().head_block_num}
        self.output_cfg = {'NUM_PARAMS_MAP':params_num-cam_dim, 'NUM_CENTER_MAP':1, 'NUM_CAM_MAP':cam_dim}

        # GLOBAL module
        self.l_final_layers = self._make_final_layers(self.backbone.backbone_channels)
        self.r_final_layers = self._make_final_layers(self.backbone.backbone_channels)

        # PART module
        self.contact_layers = self._make_contact_layers(self.backbone.backbone_channels)
        self.cam_shape_layers = self._make_cam_shape_layers()
        self.segmentation_layers = self._make_segmentation_layers(self.backbone.backbone_channels)

        self.coordmaps = get_coord_maps(128)

    def _make_final_layers(self, input_channels):
        final_layers = []
        final_layers.append(None)

        input_channels += 2
        if args().merge_mano_camera_head:
            print('Merging head not applicable')
            raise NotImplementedError
            final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_PARAMS_MAP']+self.output_cfg['NUM_CAM_MAP']))
            final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_CENTER_MAP']))
        else:
            final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_PARAMS_MAP']))
            final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_CENTER_MAP']))
            final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_CAM_MAP']))
            if args().inter_prior:
                final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_PARAMS_MAP']))

        return nn.ModuleList(final_layers)

    def _make_cam_shape_layers(self):
        final_layers = []
        final_layers.append(None)
        final_layers.append(nn.Sequential(
                    nn.Conv2d(
                        in_channels=256,
                        out_channels=64,
                        kernel_size=1,
                        stride=1,
                        padding=0)))

        # shape layers
        final_layers.append(nn.Linear(64*16, 10))
        final_layers.append(nn.Linear(64*16, 10))

        return nn.ModuleList(final_layers)

    def _make_contact_layers(self, input_channels):
        final_layers = []
        final_layers.append(None)

        input_channels += 2
        # deconv 1
        final_layers.append(nn.Sequential(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=256,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)))

        # part head
        final_layers.append(LocallyConnected2d(
            in_channels=256,
            out_channels=6,
            output_size=[16, 1],
            kernel_size=1,
            stride=1,
        ))
        final_layers.append(LocallyConnected2d(
            in_channels=256,
            out_channels=6,
            output_size=[16, 1],
            kernel_size=1,
            stride=1,
        ))

        # if concat, add 2 more layer
        if args().offset_mode == 'concat':
            final_layers.append(nn.Conv2d(in_channels=2 * 109, out_channels=109,\
            kernel_size=1,stride=1,padding=0))
            final_layers.append(nn.Conv2d(in_channels=2 * 109, out_channels=109,\
            kernel_size=1,stride=1,padding=0))

        return nn.ModuleList(final_layers)

    def _make_segmentation_layers(self, input_channels):
        final_layers = []
        final_layers.append(None)

        input_channels += 2
        # deconv2
        final_layers.append(nn.Sequential(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=256,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                    nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)))

        final_layers.append(nn.Sequential(
                nn.Conv2d(
                    in_channels=256,
                    out_channels=33,
                    kernel_size=1,
                    stride=1,
                    padding=0)))

        return nn.ModuleList(final_layers)

    def _make_head_layers(self, input_channels, output_channels): # from [B, C, H, W] to [B, aim_c, H/2, W/2] 
        head_layers = []
        num_channels = self.head_cfg['NUM_CHANNELS']

        kernel_sizes, strides, paddings = self._get_trans_cfg()
        for kernel_size, padding, stride in zip(kernel_sizes, paddings, strides):
            head_layers.append(nn.Sequential(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding),
                    nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)))
        
        for i in range(self.head_cfg['NUM_HEADS']):
            layers = []
            for _ in range(self.head_cfg['NUM_BASIC_BLOCKS']):
                layers.append(nn.Sequential(BasicBlock(num_channels, num_channels)))
            head_layers.append(nn.Sequential(*layers))

        head_layers.append(nn.Conv2d(in_channels=num_channels,out_channels=output_channels,\
            kernel_size=1,stride=1,padding=0))

        return nn.Sequential(*head_layers)

    def _get_trans_cfg(self):
        if self.outmap_size == 32:
            kernel_sizes = [3,3]
            paddings = [1,1]
            strides = [2,2]
        elif self.outmap_size == 64:
            kernel_sizes = [3]
            paddings = [1]
            strides = [2]
        elif self.outmap_size == 128:
            kernel_sizes = [3]
            paddings = [1]
            strides = [1]

        return kernel_sizes, strides, paddings



########################################################
# End of ACR arch.
########################################################

##############
# Coord Conv
##############
def get_coord_maps(size=128):
    xx_ones = torch.ones([1, size], dtype=torch.int32)
    xx_ones = xx_ones.unsqueeze(-1)

    xx_range = torch.arange(size, dtype=torch.int32).unsqueeze(0)
    xx_range = xx_range.unsqueeze(1)

    xx_channel = torch.matmul(xx_ones, xx_range)
    xx_channel = xx_channel.unsqueeze(-1)

    yy_ones = torch.ones([1, size], dtype=torch.int32)
    yy_ones = yy_ones.unsqueeze(1)

    yy_range = torch.arange(size, dtype=torch.int32).unsqueeze(0)
    yy_range = yy_range.unsqueeze(-1)

    yy_channel = torch.matmul(yy_range, yy_ones)
    yy_channel = yy_channel.unsqueeze(-1)

    xx_channel = xx_channel.permute(0, 3, 1, 2)
    yy_channel = yy_channel.permute(0, 3, 1, 2)

    xx_channel = xx_channel.float() / (size - 1)
    yy_channel = yy_channel.float() / (size - 1)

    xx_channel = xx_channel * 2 - 1
    yy_channel = yy_channel * 2 - 1

    out = torch.cat([xx_channel, yy_channel], dim=1)
    return out

################
# part-seg net
################
class SegmHead(nn.Module):
    def __init__(self, in_dim, hidden_dim1, hidden_dim2, class_dim):
        super().__init__()

        # upsample features
        self.upsampler = UpSampler(in_dim, hidden_dim1, hidden_dim2)

        segm_net = DoubleConv(hidden_dim2, class_dim)
        segm_net.double_conv = segm_net.double_conv[:4]
        self.segm_net = segm_net

    def forward(self, img_feat):
        # feature up sample to 256
        hr_img_feat = self.upsampler(img_feat)
        segm_logits = self.segm_net(hr_img_feat)
        return {'segm_logits': segm_logits}


class SegmNet(nn.Module):
    def __init__(self, out_dim):
        super(SegmNet, self).__init__()
        self.segm_head = SegmHead(32, 128, 64, out_dim)

    def map2labels(self, segm_hand):
        with torch.no_grad():
            segm_hand = segm_hand.permute(0, 2, 3, 1)
            _, pred_segm_hand = segm_hand.max(dim=3)
            return pred_segm_hand

    def forward(self, img_feat):
        segm_dict = self.segm_head(img_feat)

        segm_logits = segm_dict['segm_logits']
        #segm_mask = self.map2labels(segm_logits)
        #image = np.stack((segm_mask[0].cpu().numpy(), segm_mask[0].cpu().numpy(), segm_mask[0].cpu().numpy()), axis=2)

        return segm_logits #, segm_mask

#################
# basic modules
#################
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            #self.up = nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=True)
            #self.up = F.interpolate(scale_factor=(2, 2), mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1):
        if self.bilinear:
            x1 = F.interpolate(x1, scale_factor=(2, 2), mode='bilinear', align_corners=True)
        else:
            x1 = self.up(x1)
        return self.conv(x1)

class UpSampler(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super().__init__()
        self.up1 = Up(in_dim, out_dim)

    def forward(self, x):
        x = self.up1(x)
        return x

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, BN=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BN(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class LocallyConnected2d(nn.Module):
    # PARE head
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, bias=False):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size ** 2),
            requires_grad=True,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1]), requires_grad=True
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                               momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        nn.BatchNorm2d(num_inchannels[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3),
                                nn.ReLU(True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse

######################
# hrnet
######################
class HigherResolutionNet(nn.Module):

    def __init__(self, **kwargs):
        self.inplanes = 64
        super(HigherResolutionNet, self).__init__()
        self.make_baseline()
        self.backbone_channels = 32
        if 'part' in args().attention_mode:
            self.hand_segm = SegmNet(out_dim=33)
        else:
            raise ValueError

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1,BN=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,BN=BN))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,BN=BN))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def make_baseline(self):
        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4, BN=nn.BatchNorm2d)

        self.stage2_cfg = {'NUM_MODULES': 1, 'NUM_BRANCHES': 2, 'BLOCK': 'BASIC',\
            'NUM_BLOCKS': [4,4], 'NUM_CHANNELS':[32,64], 'FUSE_METHOD': 'SUM'}
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = {'NUM_MODULES': 4, 'NUM_BRANCHES': 3, 'BLOCK': 'BASIC',\
            'NUM_BLOCKS': [4,4,4], 'NUM_CHANNELS':[32,64,128], 'FUSE_METHOD': 'SUM'}
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = {'NUM_MODULES': 3, 'NUM_BRANCHES': 4, 'BLOCK': 'BASIC',\
            'NUM_BLOCKS': [4,4,4,4], 'NUM_CHANNELS':[32,64,128,256], 'FUSE_METHOD': 'SUM'}
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=False)

    def forward(self, x):
        x = ((BHWC_to_BCHW(x)/ 255.) * 2.0 - 1.0).contiguous()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        x = y_list[0]
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}
