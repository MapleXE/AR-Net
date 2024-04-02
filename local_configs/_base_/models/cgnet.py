# model settings
norm_cfg = dict(type='BN', eps=1e-03, requires_grad=True)
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
        type='FCNHead',
        in_channels=256,
        in_index=2,
        channels=256,
        num_convs=0,
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
