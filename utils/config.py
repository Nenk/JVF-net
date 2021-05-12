import os

configure = {
    'network': dict(
        type='resnet',
        class_num=24),

    'training': dict(
        start_epoch=0,
        start_iteration=0,
        batch_size=32,
        max_epoch=200,
        lr=0.02,
        momentum=0.9,
        weight_decay=0.0001,
        gamma=0.9,  # "lr_policy: step"
        step_size=1000,  # "lr_policy: step"
        interval_validate=1000,
    ),
    'csv_list': '../dataset/RAVDESS/RAVDESS_image.csv',
    'log_dir': '../log/',
    'checkpoint_dir': '../saved',
}