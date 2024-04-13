_base_ = [
    '../_base_/models/segformer.py',
    # '../_base_/datasets/rellis_g5.py',
    '../_base_/datasets/rugd_g5.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k_SGD.py'
]

# model settings
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/mit_b1.pth',
    backbone=dict(
        type='mit_b1',
        style='pytorch'),
    decode_head=dict(
        type='SegFormerAtt',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=256,
        dropout_ratio=0.1,
        num_classes=5,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=250),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        img_size = (375,600)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# data
data = dict(samples_per_gpu=1)
# evaluation = dict(interval=4000, metric='mIoU')

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=4e-5)
# optimizer_config = dict()
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, warmup='linear',
#     warmup_iters=1500,
#     warmup_ratio=1e-6,
#     by_epoch=False)