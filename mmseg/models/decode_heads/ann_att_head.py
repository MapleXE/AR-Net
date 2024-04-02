import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import HEADS
from ..utils import SelfAttentionBlock as _SelfAttentionBlock
from .decode_head import BaseDecodeHead

import numpy as np
from ..losses import accuracy
from ..backbones.mix_transformer import Attention
from ..builder import build_loss
from mmseg.ops import resize
class PPMConcat(nn.ModuleList):
    """Pyramid Pooling Module that only concat the features of each layer.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
    """

    def __init__(self, pool_scales=(1, 3, 6, 8)):
        super(PPMConcat, self).__init__(
            [nn.AdaptiveAvgPool2d(pool_scale) for pool_scale in pool_scales])

    def forward(self, feats):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(feats)
            ppm_outs.append(ppm_out.view(*feats.shape[:2], -1))
        concat_outs = torch.cat(ppm_outs, dim=2)
        return concat_outs


class SelfAttentionBlock(_SelfAttentionBlock):
    """Make a ANN used SelfAttentionBlock.

    Args:
        low_in_channels (int): Input channels of lower level feature,
            which is the key feature for self-attention.
        high_in_channels (int): Input channels of higher level feature,
            which is the query feature for self-attention.
        channels (int): Output channels of key/query transform.
        out_channels (int): Output channels.
        share_key_query (bool): Whether share projection weight between key
            and query projection.
        query_scale (int): The scale of query feature map.
        key_pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module of key feature.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict|None): Config of activation layers.
    """

    def __init__(self, low_in_channels, high_in_channels, channels,
                 out_channels, share_key_query, query_scale, key_pool_scales,
                 conv_cfg, norm_cfg, act_cfg):
        key_psp = PPMConcat(key_pool_scales)
        if query_scale > 1:
            query_downsample = nn.MaxPool2d(kernel_size=query_scale)
        else:
            query_downsample = None
        super(SelfAttentionBlock, self).__init__(
            key_in_channels=low_in_channels,
            query_in_channels=high_in_channels,
            channels=channels,
            out_channels=out_channels,
            share_key_query=share_key_query,
            query_downsample=query_downsample,
            key_downsample=key_psp,
            key_query_num_convs=1,
            key_query_norm=True,
            value_out_num_convs=1,
            value_out_norm=False,
            matmul_norm=True,
            with_out=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)


class AFNB(nn.Module):
    """Asymmetric Fusion Non-local Block(AFNB)

    Args:
        low_in_channels (int): Input channels of lower level feature,
            which is the key feature for self-attention.
        high_in_channels (int): Input channels of higher level feature,
            which is the query feature for self-attention.
        channels (int): Output channels of key/query transform.
        out_channels (int): Output channels.
            and query projection.
        query_scales (tuple[int]): The scales of query feature map.
            Default: (1,)
        key_pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module of key feature.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict|None): Config of activation layers.
    """

    def __init__(self, low_in_channels, high_in_channels, channels,
                 out_channels, query_scales, key_pool_scales, conv_cfg,
                 norm_cfg, act_cfg):
        super(AFNB, self).__init__()
        self.stages = nn.ModuleList()
        for query_scale in query_scales:
            self.stages.append(
                SelfAttentionBlock(
                    low_in_channels=low_in_channels,
                    high_in_channels=high_in_channels,
                    channels=channels,
                    out_channels=out_channels,
                    share_key_query=False,
                    query_scale=query_scale,
                    key_pool_scales=key_pool_scales,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        self.bottleneck = ConvModule(
            out_channels + high_in_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

    def forward(self, low_feats, high_feats):
        """Forward function."""
        priors = [stage(high_feats, low_feats) for stage in self.stages]
        context = torch.stack(priors, dim=0).sum(dim=0)
        output = self.bottleneck(torch.cat([context, high_feats], 1))
        return output


class APNB(nn.Module):
    """Asymmetric Pyramid Non-local Block (APNB)

    Args:
        in_channels (int): Input channels of key/query feature,
            which is the key feature for self-attention.
        channels (int): Output channels of key/query transform.
        out_channels (int): Output channels.
        query_scales (tuple[int]): The scales of query feature map.
            Default: (1,)
        key_pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module of key feature.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict|None): Config of activation layers.
    """

    def __init__(self, in_channels, channels, out_channels, query_scales,
                 key_pool_scales, conv_cfg, norm_cfg, act_cfg):
        super(APNB, self).__init__()
        self.stages = nn.ModuleList()
        for query_scale in query_scales:
            self.stages.append(
                SelfAttentionBlock(
                    low_in_channels=in_channels,
                    high_in_channels=in_channels,
                    channels=channels,
                    out_channels=out_channels,
                    share_key_query=True,
                    query_scale=query_scale,
                    key_pool_scales=key_pool_scales,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        self.bottleneck = ConvModule(
            2 * in_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, feats):
        """Forward function."""
        priors = [stage(feats, feats) for stage in self.stages]
        context = torch.stack(priors, dim=0).sum(dim=0)
        output = self.bottleneck(torch.cat([context, feats], 1))
        return output


@HEADS.register_module()
class ANNAttHead(BaseDecodeHead):
    """Asymmetric Non-local Neural Networks for Semantic Segmentation.

    This head is the implementation of `ANNNet
    <https://arxiv.org/abs/1908.07678>`_.

    Args:
        project_channels (int): Projection channels for Nonlocal.
        query_scales (tuple[int]): The scales of query feature map.
            Default: (1,)
        key_pool_scales (tuple[int]): The pooling scales of key feature map.
            Default: (1, 3, 6, 8).
    """

    def __init__(self,
                 project_channels,
                 img_size,
                 query_scales=(1, ),
                 key_pool_scales=(1, 3, 6, 8),
                 **kwargs):
        super(ANNAttHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        assert len(self.in_channels) == 2
        low_in_channels, high_in_channels = self.in_channels
        self.project_channels = project_channels
        self.fusion = AFNB(
            low_in_channels=low_in_channels,
            high_in_channels=high_in_channels,
            out_channels=high_in_channels,
            channels=project_channels,
            query_scales=query_scales,
            key_pool_scales=key_pool_scales,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.bottleneck = ConvModule(
            high_in_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.context = APNB(
            in_channels=self.channels,
            out_channels=self.channels,
            channels=project_channels,
            query_scales=query_scales,
            key_pool_scales=key_pool_scales,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        
        self.h = int(np.ceil(img_size[0]/8))
        self.w = int(np.ceil(img_size[1]/8))

        self.attn = Attention(dim=self.channels,num_heads=self.num_classes,h=self.h, w=self.w)
        self.attn_loss = build_loss(dict(
                     type='ClassAttCrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0, num_classes=self.num_classes))


    def forward(self, inputs):
        """Forward function."""
        low_feats, high_feats = self._transform_inputs(inputs)
        output = self.fusion(low_feats, high_feats)
        output = self.dropout(output)
        output = self.bottleneck(output)
        output = self.context(output)

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

