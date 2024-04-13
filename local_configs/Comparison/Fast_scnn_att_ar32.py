_base_ = [
    '../_base_/models/fast_scnn.py',
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
    backbone=dict(
        type='FastSCNN',
        downsample_dw_channels=(32, 48),
        global_in_channels=64,
        global_block_channels=(64, 96, 128),
        global_block_strides=(2, 2, 1),
        global_out_channels=128,
        higher_in_channels=64,
        lower_in_channels=128,
        fusion_out_channels=128,
        out_indices=(0, 1, 2),
        norm_cfg=norm_cfg,
        align_corners=False),
    decode_head=dict(
        type='FCNSepAttHead',
        img_size = (375,600),
        AR_Ratio = 32,
        in_channels=128,
        channels=128,
        concat_input=False,
        num_classes=4,
        in_index=-1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


# data
data = dict(samples_per_gpu=2)
# evaluation = dict(interval=4000, metric='mIoU')

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