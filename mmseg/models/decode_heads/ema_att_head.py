import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .decode_head import BaseDecodeHead

import numpy as np
from ..losses import accuracy
from ..backbones.mix_transformer import Attention
from ..builder import build_loss
from mmseg.ops import resize

def reduce_mean(tensor):
    """Reduce mean when distributed training."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


class EMAModule(nn.Module):
    """Expectation Maximization Attention Module used in EMANet.

    Args:
        channels (int): Channels of the whole module.
        num_bases (int): Number of bases.
        num_stages (int): Number of the EM iterations.
    """

    def __init__(self, channels, num_bases, num_stages, momentum):
        super(EMAModule, self).__init__()
        assert num_stages >= 1, 'num_stages must be at least 1!'
        self.num_bases = num_bases
        self.num_stages = num_stages
        self.momentum = momentum

        bases = torch.zeros(1, channels, self.num_bases)
        bases.normal_(0, math.sqrt(2. / self.num_bases))
        # [1, channels, num_bases]
        bases = F.normalize(bases, dim=1, p=2)
        self.register_buffer('bases', bases)

    def forward(self, feats):
        """Forward function."""
        batch_size, channels, height, width = feats.size()
        # [batch_size, channels, height*width]
        feats = feats.view(batch_size, channels, height * width)
        # [batch_size, channels, num_bases]
        bases = self.bases.repeat(batch_size, 1, 1)

        with torch.no_grad():
            for i in range(self.num_stages):
                # [batch_size, height*width, num_bases]
                attention = torch.einsum('bcn,bck->bnk', feats, bases)
                attention = F.softmax(attention, dim=2)
                # l1 norm
                attention_normed = F.normalize(attention, dim=1, p=1)
                # [batch_size, channels, num_bases]
                bases = torch.einsum('bcn,bnk->bck', feats, attention_normed)
                # l2 norm
                bases = F.normalize(bases, dim=1, p=2)

        feats_recon = torch.einsum('bck,bnk->bcn', bases, attention)
        feats_recon = feats_recon.view(batch_size, channels, height, width)

        if self.training:
            bases = bases.mean(dim=0, keepdim=True)
            bases = reduce_mean(bases)
            # l2 norm
            bases = F.normalize(bases, dim=1, p=2)
            self.bases = (1 -
                          self.momentum) * self.bases + self.momentum * bases

        return feats_recon


@HEADS.register_module()
class EMAAttHead(BaseDecodeHead):
    """Expectation Maximization Attention Networks for Semantic Segmentation.

    This head is the implementation of `EMANet
    <https://arxiv.org/abs/1907.13426>`_.

    Args:
        ema_channels (int): EMA module channels
        num_bases (int): Number of bases.
        num_stages (int): Number of the EM iterations.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer. Default: True
        momentum (float): Momentum to update the base. Default: 0.1.
    """

    def __init__(self,
                 ema_channels,
                 img_size,
                 num_bases,
                 num_stages,
                 AR_Ratio=8,
                 concat_input=True,
                 momentum=0.1,
                 **kwargs):
        super(EMAAttHead, self).__init__(**kwargs)
        self.ema_channels = ema_channels
        self.num_bases = num_bases
        self.num_stages = num_stages
        self.concat_input = concat_input
        self.momentum = momentum
        self.ema_module = EMAModule(self.ema_channels, self.num_bases,
                                    self.num_stages, self.momentum)

        self.ema_in_conv = ConvModule(
            self.in_channels,
            self.ema_channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # project (0, inf) -> (-inf, inf)
        self.ema_mid_conv = ConvModule(
            self.ema_channels,
            self.ema_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=None,
            act_cfg=None)
        for param in self.ema_mid_conv.parameters():
            param.requires_grad = False

        self.ema_out_conv = ConvModule(
            self.ema_channels,
            self.ema_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.bottleneck = ConvModule(
            self.ema_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            
        
        self.AR_Ratio = AR_Ratio    
        self.h = int(np.ceil(img_size[0]/AR_Ratio))
        self.w = int(np.ceil(img_size[1]/AR_Ratio))

        self.attn = Attention(dim=self.channels,num_heads=self.num_classes,h=self.h, w=self.w)
        self.attn_loss = build_loss(dict(
                     type='ClassAttCrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0, num_classes=self.num_classes))

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        feats = self.ema_in_conv(x)
        identity = feats
        feats = self.ema_mid_conv(feats)
        recon = self.ema_module(feats)
        recon = F.relu(recon, inplace=True)
        recon = self.ema_out_conv(recon)
        output = F.relu(identity + recon, inplace=True)
        output = self.bottleneck(output)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        
        output_compressed = resize(output, size=[self.h,self.w],mode='bilinear',align_corners=False)
        attn_additional = self.attn.forward_additional(output_compressed,self.h,self.w)
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

