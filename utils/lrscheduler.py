import math
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):

    def __init__(self, optimizer, warmup_epochs, max_lr, max_epochs, min_lr=0.0, last_epoch=-1, verbose=False):
        self.warmup_epochs = warmup_epochs
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.max_epochs = max_epochs
        self.period = self.max_epochs - self.warmup_epochs

        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        # inside the warmup epochs
        if self.last_epoch < self.warmup_epochs:
            curr_lr = self.min_lr + (self.max_lr - self.min_lr) * self.last_epoch / self.warmup_epochs
        # cosine schedule after warmup
        else:
            curr_epochs = self.last_epoch - self.warmup_epochs
            curr_lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                    1 + math.cos(math.pi * curr_epochs / self.period)
            )

        return [curr_lr for group in self.optimizer.param_groups]
