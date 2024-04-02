_base_ = [
    '../_base_/models/gcnet_r50-d8.py',
    # '../_base_/datasets/rellis_g5.py',
    '../_base_/datasets/rugd_g5.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k_SGD.py'
]

# model settings
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True

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