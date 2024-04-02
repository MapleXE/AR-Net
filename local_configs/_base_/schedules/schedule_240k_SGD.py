# optimizer
optimizer = dict(type='SGD', lr=0.003, momentum=0.9,weight_decay=0.00005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=240000)
checkpoint_config = dict(by_epoch=False, interval=16000)
# evaluation = dict(interval=4000, metric='mIoU')
