import torch
import lightning as L
from torch.nn import CrossEntropyLoss
from torchmetrics.functional import accuracy
from torchmetrics.classification import MulticlassConfusionMatrix
from matplotlib import pyplot as plt
from model import Network
import config


class LitModel(L.LightningModule):
    """
    Pytorch Lightning Model，用于训练、验证、测试，以及生成模型。
    参数来自于config.py，修改时确保仅在config.py中修改。
    多数参数在此处无用，仅用于保存超参数。
    """

    def __init__(
        self,
        seed=config.SEED,
        num_epochs=config.NUM_EPOCHS,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        learning_rate=config.LEARNING_RATE,
        in_channel=config.IN_CHANNEL,
        num_classes=config.NUM_CLASSES,
        task=config.TASK,
        sch_patience=config.SCH_PATIENCE,
    ):
        super().__init__()
        # 初始化所需超参数
        self.learning_rate = learning_rate
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.task = task
        self.sch_patience = sch_patience
        self.save_hyperparameters()  # 保存参数

        # 初始化模型、损失函数和优化器
        self.model = Network(784, 256, 10)
        self.criterion = CrossEntropyLoss()

        # 初始化训练、验证、测试的输出
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    # 用于生成模型
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        训练步骤，计算训练集上的损失和准确率。

        Args:
            必须设置为batch和batch_idx，否则无法进行反向传播。
            batch: 输入的批次数据。
            batch_idx: 批次索引。

        Returns:
            loss: 训练集上的损失。必须返回loss，否则无法进行反向传播。

        """
        # 主要步骤
        imgs, labels = batch  # 从批次中获取图像和标签
        preds = self.model(imgs)  # 通过模型获取预测值
        loss = self.criterion(preds, labels)  # 计算损失
        acc = accuracy(preds, labels, self.task, num_classes=self.num_classes)  # 计算准确率

        # 将训练过程中的损失和准确率记录到tensorboard日志中
        tensorboard_logs = {"train_loss": loss, "train_acc": acc}
        self.training_step_outputs.append({"loss": loss, "acc": acc, "log": tensorboard_logs})
        return loss  # 返回损失（必须返回loss，否则无法进行反向传播）

    def on_train_epoch_end(self):
        """
        训练周期结束时，计算平均损失和准确率。

        """
        avg_loss = torch.stack([x["loss"] for x in self.training_step_outputs]).mean()  # 计算平均损失
        train_acc = torch.stack([x["acc"] for x in self.training_step_outputs]).mean()  # 计算平均准确率

        # 将训练过程中的损失、准确率和步骤记录到tensorboard日志中
        tensorboard_logs = {"train_loss": avg_loss, "train_acc": train_acc, "step": self.trainer.current_epoch + 1}
        self.log_dict(tensorboard_logs)  # 将日志字典保存到Tensorbboard中

    # 验证步骤，计算验证集上的损失和准确率
    def validation_step(self, batch, batch_idx):
        """
        验证步骤，计算验证集上的损失和准确率。

        Args:
            必须设置为batch和batch_idx，否则无法进行反向传播。
            batch: 输入的批次数据。
            batch_idx: 批次索引。

        """
        # 主要步骤
        imgs, labels = batch  # 从批次中获取图像和标签
        preds = self.model(imgs)  # 通过模型获取预测值
        loss = self.criterion(preds, labels)  # 计算损失
        acc = accuracy(preds, labels, self.task, num_classes=self.num_classes)  # 计算准确率

        # 将验证过程中的损失和准确率记录到tensorboard日志中
        tensorboard_logs = {"val_loss": loss, "val_acc": acc}
        self.validation_step_outputs.append({"loss": loss, "acc": acc, "log": tensorboard_logs})

    # 验证周期结束时，计算平均损失和准确率
    def on_validation_epoch_end(self):
        """
        验证周期结束时，计算平均损失和准确率。

        """
        avg_loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()  # 计算平均损失
        val_acc = torch.stack([x["acc"] for x in self.validation_step_outputs]).mean()  # 计算平均准确率

        # 将验证过程中的损失、准确率和步骤记录到tensorboard日志中
        tensorboard_logs = {"val_loss": avg_loss, "val_acc": val_acc, "step": self.trainer.current_epoch + 1}
        self.log_dict(tensorboard_logs)  # 将日志字典保存到Tensorbboard中

    def test_step(self, batch, batch_idx):
        """
        测试步骤，计算测试集上的损失和准确率。

        Args:
            必须设置为batch和batch_idx，否则无法进行反向传播。
            batch: 输入的批次数据。
            batch_idx: 批次索引。

        """
        # 主要步骤
        imgs, labels = batch  # 从批次中获取图像和标签
        preds = self.model(imgs)  # 通过模型获取预测值
        loss = self.criterion(preds, labels)  # 计算损失
        acc = accuracy(preds, labels, self.task, num_classes=self.num_classes)  # 计算准确率

        # 将测试过程中的损失和准确率记录到tensorboard日志中
        tensorboard_logs = {"test_loss": loss, "test_acc": acc}
        self.test_step_outputs.append(
            {"loss": loss, "acc": acc, "preds": preds, "targets": labels, "log": tensorboard_logs}
        )

    def on_test_epoch_end(self):
        """
        测试周期结束时，计算平均损失和准确率，并绘制混淆矩阵。

        """
        avg_loss = torch.stack([x["loss"] for x in self.test_step_outputs]).mean()  # 计算平均损失
        test_acc = torch.stack([x["acc"] for x in self.test_step_outputs]).mean()  # 计算平均准确率
        preds = torch.argmax(torch.cat([x["preds"] for x in self.test_step_outputs]), dim=1).cpu()  # 获取预测值
        targets = torch.argmax(torch.cat([x["targets"] for x in self.test_step_outputs]), dim=1).cpu()  # 获取标签

        # 绘制混淆矩阵
        confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes)  # 初始化混淆矩阵
        confusion_matrix.update(preds, targets)  # 更新混淆矩阵
        cm_fig, cm_ax = confusion_matrix.plot()  # 绘制混淆矩阵
        plt.close()  # 关闭绘图窗口

        # 将测试过程中的损失、准确率和混淆矩阵记录到tensorboard日志中
        self.logger.experiment.add_figure("Confusion Matrix", cm_fig)
        tensorboard_logs = {"test_loss": avg_loss, "test_acc": test_acc}
        self.log_dict(tensorboard_logs)

    def configure_optimizers(self):
        """
        用于生成模型优化器。

        Returns:
            optimizer: 生成的优化器。
            lr_scheduler: 生成的学习率调度器。
            monitor: 监控的指标。

        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)  # 生成优化器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=self.sch_patience, min_lr=5e-5, cooldown=30
        )  # 生成学习率调度器
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
