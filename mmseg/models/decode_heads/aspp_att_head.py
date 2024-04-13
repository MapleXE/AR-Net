import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead


import numpy as np
from ..losses import accuracy
from ..backbones.mix_transformer import Attention
from ..builder import build_loss

class ASPPModule(nn.ModuleList):
    """Atrous Spatial Pyramid Pooling (ASPP) Module.

    Args:
        dilations (tuple[int]): Dilation rate of each layer.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, dilations, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg):
        super(ASPPModule, self).__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for dilation in dilations:
            self.append(
                ConvModule(
                    self.in_channels,
                    self.channels,
                    1 if dilation == 1 else 3,
                    dilation=dilation,
                    padding=0 if dilation == 1 else dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        for aspp_module in self:
            aspp_outs.append(aspp_module(x))

        return aspp_outs


@HEADS.register_module()
class ASPPAttHead(BaseDecodeHead):
    """Rethinking Atrous Convolution for Semantic Image Segmentation.

    This head is the implementation of `DeepLabV3
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        dilations (tuple[int]): Dilation rates for ASPP module.
            Default: (1, 6, 12, 18).
    """

    def __init__(self, img_size, dilations=(1, 6, 12, 18), AR_Ratio=8, **kwargs):
        super(ASPPAttHead, self).__init__(**kwargs)
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                self.in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        self.aspp_modules = ASPPModule(
            dilations,
            self.in_channels,
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.bottleneck = ConvModule(
            (len(dilations) + 1) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        
        self.AR_Ratio = AR_Ratio
        self.h = int(np.ceil(img_size[0]/AR_Ratio))
        self.w = int(np.ceil(img_size[1]/AR_Ratio))

        self.attn = Attention(dim=(len(dilations) + 1) * self.channels,num_heads=self.num_classes,h=self.h, w=self.w)
        self.attn_loss = build_loss(dict(
                     type='ClassAttCrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0, num_classes=self.num_classes))
        

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)


        output_compressed = resize(aspp_outs, size=[self.h,self.w],mode='bilinear',align_corners=False)
        attn_additional = self.attn.forward_additional(output_compressed,self.h,self.w)
        attn_additional = attn_additional.permute(1, 0, 2, 3)
        attn_additional = torch.diagonal(attn_additional, dim1=2, dim2=3).view(self.num_classes, -1, self.h, self.w)

        output = self.bottleneck(aspp_outs)
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
