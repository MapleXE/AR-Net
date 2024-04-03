import torch
from mmcv.cnn import ContextBlock

from ..builder import HEADS
from .fcn_head import FCNHead

import numpy as np
from ..losses import accuracy
from ..backbones.mix_transformer import Attention
from ..builder import build_loss
from mmseg.ops import resize

@HEADS.register_module()
class GCAttHead(FCNHead):
    """GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond.

    This head is the implementation of `GCNet
    <https://arxiv.org/abs/1904.11492>`_.

    Args:
        ratio (float): Multiplier of channels ratio. Default: 1/4.
        pooling_type (str): The pooling type of context aggregation.
            Options are 'att', 'avg'. Default: 'avg'.
        fusion_types (tuple[str]): The fusion type for feature fusion.
            Options are 'channel_add', 'channel_mul'. Defautl: ('channel_add',)
    """

    def __init__(self,
                 img_size,
                 ratio=1 / 4.,
                 pooling_type='att',
                 fusion_types=('channel_add', ),
                 **kwargs):
        super(GCAttHead, self).__init__(num_convs=2, **kwargs)
        self.ratio = ratio
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        self.gc_block = ContextBlock(
            in_channels=self.channels,
            ratio=self.ratio,
            pooling_type=self.pooling_type,
            fusion_types=self.fusion_types)
        
        self.h = int(np.ceil(img_size[0]/8))
        self.w = int(np.ceil(img_size[1]/8))

        self.attn = Attention(dim=self.channels,num_heads=self.num_classes,h=self.h, w=self.w)
        self.attn_loss = build_loss(dict(
                     type='ClassAttCrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0, num_classes=self.num_classes))

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.convs[0](x)
        output = self.gc_block(output)

        attn_additional = self.attn.forward_additional(output,self.h,self.w)
        attn_additional = attn_additional.permute(1, 0, 2, 3)
        attn_additional = torch.diagonal(attn_additional, dim1=2, dim2=3).view(self.num_classes, -1, self.h, self.w)
        
        output = self.convs[1](output)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))

        
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


