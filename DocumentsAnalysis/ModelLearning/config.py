"""===============================================================================*
| Конфигурация модели. Редактируется ЗДЕСЬ!!! (численные значения, остальные нет).|
| Все значения None примут своё значение в процессе обучения.                     |
| Редактировать можно только параметры оптимизатора, шедулера и тд.               |
| Остальное может некорректно обрабатываться в случае изменения.                  |
*==============================================================================="""
config = {
    "run_path": None,
    "model": {
        "architecture": "resnet50",
        "base_weights": "IMAGENET1K_V1",
        "unfrozen_layers": ["layer4", "fc"],
        "num_classes": 2,
        "input_size": [3, 224, 224],
        "device": None,
        "prev_epochs": 0,
        "add_epochs": 15,

        "loss_fn": "CrossEntropyLoss",
        "optimizer": {
            "type": "AdamW",
            "lr": 0.001,
            "betas": [0.9, 0.999],
            "weight_decay": 0.01,
        },
        "scheduler": {
            "type": "ReduceLROnPlateau",
            "mode": "min",
            "factor": 0.5,
            "patience": 2,
            "threshold": 1e-4,
        },
    },
    "training": {
        "batch_size": 32,
        "use_amp": None,
        "cuDNN": False,
    },
    "dataset": {
        "train_forged_path": "./dataset_patches/training_set/forged",
        "train_genuine_path": "./dataset_patches/training_set/genuine",
        "test_forged_path": "./dataset_patches/test_set/forged",
        "test_genuine_path": "./dataset_patches/test_set/genuine",
        "validation_forged_path": "./dataset_patches/validation_set/forged",
        "validation_genuine_path": "./dataset_patches/validation_set/genuine",
        "class_names": ["genuine", "forged"],
        "augmentation": True,
        "train_size": None,
        "val_size": None,
        "test_size": None,
    },
}
