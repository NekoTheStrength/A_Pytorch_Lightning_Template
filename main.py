import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torchvision.transforms import Compose, ToTensor
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from lit import LitModel
import config

import warnings

warnings.filterwarnings("ignore")  # 忽略警告


def loader(seed, batch_size=32, num_workers=8):
    """dataloader加载函数示例，在使用时，在utils.py中（或者你自己的文件中）实现你的加载函数。"""
    if num_workers > 0:
        persistent_workers = True
    else:
        persistent_workers = False
    transform = Compose([ToTensor()])
    train_set = MNIST("./data", download=True, train=True, transform=transform)
    test_set = MNIST("./data", download=True, train=False, transform=transform)
    train, val = random_split(train_set, [0.8, 0.2], generator=seed)
    train = DataLoader(
        train, batch_size=batch_size, num_workers=num_workers, persistent_workers=persistent_workers, shuffle=True
    )
    val = DataLoader(val, batch_size=batch_size, num_workers=num_workers, persistent_workers=persistent_workers)
    test = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, persistent_workers=persistent_workers)
    return train, val, test


def main():
    # 设定种子
    seed = torch.Generator().manual_seed(config.SEED)
    L.seed_everything(seed=config.SEED)
    train, val, test = loader(seed, config.BATCH_SIZE, config.NUM_WORKERS)
    # 初始化模型
    model = LitModel()
    # 训练模型
    logger = TensorBoardLogger(f"./lightning_logs/Seed[{config.SEED}]", name=f"your_model_name")
    callbacks = [ModelCheckpoint(monitor="val_loss", mode="min")]
    trainer = L.Trainer(max_epochs=config.NUM_EPOCHS, logger=logger, callbacks=callbacks, precision="16-mixed")
    trainer.fit(model, train, val)
    # 测试模型
    trainer.test(model, test, ckpt_path="best")


if __name__ == "__main__":
    main()
