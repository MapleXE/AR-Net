_base_ = [
    '../_base_/models/cgnet.py',
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
        type='CGNet',
        norm_cfg=norm_cfg,
        in_channels=3,
        num_channels=(32, 64, 128),
        num_blocks=(3, 21),
        dilations=(2, 4),
        reductions=(8, 16)),
    decode_head=dict(
        type='FCNAttHead',
        in_channels=256,
        in_index=2,
        channels=256,
        num_convs=0,
        img_size = (375,600),
        concat_input=False,
        dropout_ratio=0,
        num_classes=4,
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            )),
    # model training and testing settings
    train_cfg=dict(sampler=None),
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