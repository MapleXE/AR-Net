import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer

from ..builder import HEADS
from .decode_head import BaseDecodeHead

import numpy as np
from ..losses import accuracy
from ..backbones.mix_transformer import Attention
from ..builder import build_loss
from mmseg.ops import resize
class DCM(nn.Module):
    """Dynamic Convolutional Module used in DMNet.

    Args:
        filter_size (int): The filter size of generated convolution kernel
            used in Dynamic Convolutional Module.
        fusion (bool): Add one conv to fuse DCM output feature.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict | None): Config of conv layers.
        norm_cfg (dict | None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, filter_size, fusion, in_channels, channels, conv_cfg,
                 norm_cfg, act_cfg):
        super(DCM, self).__init__()
        self.filter_size = filter_size
        self.fusion = fusion
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.filter_gen_conv = nn.Conv2d(self.in_channels, self.channels, 1, 1,
                                         0)

        self.input_redu_conv = ConvModule(
            self.in_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        if self.norm_cfg is not None:
            self.norm = build_norm_layer(self.norm_cfg, self.channels)[1]
        else:
            self.norm = None
        self.activate = build_activation_layer(self.act_cfg)

        if self.fusion:
            self.fusion_conv = ConvModule(
                self.channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def forward(self, x):
        """Forward function."""
        generted_filter = self.filter_gen_conv(
            F.adaptive_avg_pool2d(x, self.filter_size))
        x = self.input_redu_conv(x)
        b, c, h, w = x.shape
        # [1, b * c, h, w], c = self.channels
        x = x.view(1, b * c, h, w)
        # [b * c, 1, filter_size, filter_size]
        generted_filter = generted_filter.view(b * c, 1, self.filter_size,
                                               self.filter_size)
        pad = (self.filter_size - 1) // 2
        if (self.filter_size - 1) % 2 == 0:
            p2d = (pad, pad, pad, pad)
        else:
            p2d = (pad + 1, pad, pad + 1, pad)
        x = F.pad(input=x, pad=p2d, mode='constant', value=0)
        # [1, b * c, h, w]
        output = F.conv2d(input=x, weight=generted_filter, groups=b * c)
        # [b, c, h, w]
        output = output.view(b, c, h, w)
        if self.norm is not None:
            output = self.norm(output)
        output = self.activate(output)

        if self.fusion:
            output = self.fusion_conv(output)

        return output


@HEADS.register_module()
class DMAttHead(BaseDecodeHead):
    """Dynamic Multi-scale Filters for Semantic Segmentation.

    This head is the implementation of
    `DMNet <https://openaccess.thecvf.com/content_ICCV_2019/papers/\
        He_Dynamic_Multi-Scale_Filters_for_Semantic_Segmentation_\
            ICCV_2019_paper.pdf>`_.

    Args:
        filter_sizes (tuple[int]): The size of generated convolutional filters
            used in Dynamic Convolutional Module. Default: (1, 3, 5, 7).
        fusion (bool): Add one conv to fuse DCM output feature.
    """

    def __init__(self, img_size, filter_sizes=(1, 3, 5, 7), fusion=False, **kwargs):
        super(DMAttHead, self).__init__(**kwargs)
        assert isinstance(filter_sizes, (list, tuple))
        self.filter_sizes = filter_sizes
        self.fusion = fusion
        dcm_modules = []
        for filter_size in self.filter_sizes:
            dcm_modules.append(
                DCM(filter_size,
                    self.fusion,
                    self.in_channels,
                    self.channels,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        self.dcm_modules = nn.ModuleList(dcm_modules)
        self.bottleneck = ConvModule(
            self.in_channels + len(filter_sizes) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.h = int(np.ceil(img_size[0]/8))
        self.w = int(np.ceil(img_size[1]/8))

        self.attn = Attention(dim=self.channels,num_heads=self.num_classes,h=self.h, w=self.w)
        self.attn_loss = build_loss(dict(
                     type='ClassAttCrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0, num_classes=self.num_classes))

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        dcm_outs = [x]
        for dcm_module in self.dcm_modules:
            dcm_outs.append(dcm_module(x))
        dcm_outs = torch.cat(dcm_outs, dim=1)
        output = self.bottleneck(dcm_outs)

        attn_additional = self.attn.forward_additional(output,self.h,self.w)
        attn_additional = attn_additional.permute(1, 0, 2, 3)
        attn_additional = torch.diagonal(attn_additional, dim1=2, dim2=3).view(self.num_classes, -1, self.h, self.w)

        output = self.cls_seg(output)
        return output, attn_additional


    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
            """Forward function for training.
            Args:
                inputs (list[Tensor]): List of multi-level img features.
                img_metas (list[dict]): List of image info dict where each dict
                    has: 'img_shape', 'scale_factor', 'flip', and may also contain
                    'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                    For details on the values of these keys see
                    `mmseg/datasets/pipelines/formatting.py:Collect`.
                gt_semantic_seg (Tensor): Semantic segmentation masks
                    used if the architecture supports semantic segmentation task.
                train_cfg (dict): The training config.

            Returns:
                dict[str, Tensor]: a dictionary of loss components
            """
            seg_logits, att = self.forward(inputs)
            losses = self.losses([seg_logits, att], gt_semantic_seg)
            return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        out, maps = self.forward(inputs)
        # tmp = maps[0,0,:,:].cpu().numpy()
        # np.savetxt('/home/maplexe/rob599/SegFormer/demo/map.txt',tmp)
        return out

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    # @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        attn = seg_logit[1]
        attn = resize(
            input=attn,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        seg_logit = seg_logit[0]
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            # class_weight=self.class_weight,
            ignore_index=self.ignore_index)

        loss['loss_seg'] += self.attn_loss(
            attn,
            seg_label,
            weight=seg_weight,
            # class_weight=self.class_weight,
            ignore_index=self.ignore_index)



        loss['acc_seg'] = accuracy(seg_logit, seg_label)


        return loss

