from abc import ABC, abstractmethod
from typing import Optional


class EarlyStopper(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def evaluate(self, val_acc: float, val_loss: float) -> bool:
        """
        Records statistics of the last epoch and informs whether to stop training
        Args:
            val_loss: Validation loss of the last epoch
            val_acc: Validation accuracy of the last epoch

        Returns:
            A boolean. Training is continued if False and stopped if True
        """
        pass


class SimpleEarlyStopper(EarlyStopper):
    def __init__(self, patience: int = 25, 
                 val_loss_threshold: Optional[float] = 0.01,
                 val_acc_threshold: Optional[float] = 0.01,
                 patience_zero_threshold: Optional[float] = 0.999):
        self.best_val_acc: float = -1
        self.best_val_loss: float = -1
        self.wait_cnt = 0

        self.patience = patience
        self.val_loss_threshold = val_loss_threshold
        self.val_acc_threshold = val_acc_threshold
        self.patience_zero_threshold = patience_zero_threshold

    def reset(self):
        self.best_val_acc: float = -1
        self.best_val_loss: float = -1
        self.wait_cnt = 0

    def evaluate(self, val_acc: float, val_loss: float) -> bool:
        if val_acc >= self.patience_zero_threshold:
            return True

        if val_acc > self.best_val_acc + self.val_acc_threshold:
            self.best_val_acc = val_acc
            self.best_val_loss = val_loss
            self.wait_cnt = 0

        elif val_loss < (self.best_val_loss - self.val_loss_threshold):
            self.best_val_loss = val_loss
            self.wait_cnt = 0

        else:
            self.wait_cnt += 1

        return self.wait_cnt > self.patience

