import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

import os
from abc import abstractmethod
from tqdm import tqdm
from datetime import datetime
from copy import deepcopy

from data import get_dataloader
from common import Accumulator, Timer
from utils import compute_accuracy, symmetric_kl_divergence
from setup import setup_device


class Trainer:
    def __init__(self, args, model):
        self.args = args
        self.device = setup_device()
        self.model = model.to(self.device)
        self.train_loader, self.val_loader, self.test_loader = self.init_dataloader()
        self.optimizer, self.scheduler = self.load_optimizer_and_scheduler()
        self.max_accuracy = 0.0
        self.epoch_of_best_state = 0
        self.best_state = self.model.state_dict()
        self.testing_accuracy = 0.0
        self.confidence_interval = 0.0
        self.timer = Timer()
        self.accumulator = Accumulator()

    @abstractmethod
    def run(self):
        raise NotImplementedError(
            "trainer must implement a run method."
        )

    def train_epoch(self):
        self.model.train()
        self.accumulator.reset()
        with tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                desc="Training"
        ) as tqdm_train:
            for _, batch in tqdm_train:
                images, labels = [_.to(self.device) for _ in batch]
                logits = self.model(images)
                loss = self.compute_loss(logits, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.accumulator.add(loss.item())
                tqdm_train.set_postfix(loss=self.accumulator.avg())

        return self.accumulator.avg()

    def eval_epoch(self, phase=None):
        eval_loader = self.val_loader if phase == "Validation" else self.test_loader
        self.model.eval()
        self.accumulator.reset()
        with torch.no_grad():
            with tqdm(
                    enumerate(eval_loader),
                    total=len(eval_loader),
                    desc=phase,
            ) as tqdm_eval:
                for _, batch in tqdm_eval:
                    images, labels = [_.to(self.device) for _ in batch]
                    logits = self.model(images)
                    accuracy = compute_accuracy(logits, labels)
                    self.accumulator.add(accuracy)
                    tqdm_eval.set_postfix(accuracy=self.accumulator.avg())

        return self.accumulator.avg_and_confidence_interval()

    def compute_loss(self, logits, labels):
        if self.args.is_pretrained:
            return F.cross_entropy(logits, labels)
        else:
            return F.cross_entropy(logits, labels)
            # score1, score2 = logits
            # loss_ce = F.cross_entropy(score1, labels) + F.cross_entropy(score2, labels)
            # loss_kl = symmetric_kl_divergence(score1, score2)
            # return loss_ce + loss_kl

    def init_dataloader(self):
        train_loader = get_dataloader('train', self.args)
        val_loader = get_dataloader('val', self.args)
        test_loader = get_dataloader('test', self.args)
        return train_loader, val_loader, test_loader

    def load_optimizer_and_scheduler(self):
        optimizer = SGD(
            self.model.parameters(),
            lr=self.args.learning_rate,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay
        )
        scheduler = MultiStepLR(
            optimizer=optimizer,
            milestones=self.args.milestones,
            gamma=self.args.gamma,
        )
        return optimizer, scheduler

    def compare_to_best_state(self, validation_accuracy, current_epoch):
        if validation_accuracy > self.max_accuracy:
            self.max_accuracy = validation_accuracy
            self.best_state = deepcopy(self.model.state_dict())
            self.epoch_of_best_state = current_epoch
            print("Ding ding ding! We found a max accuracy models!")

    def test_model(self):
        self.testing_accuracy, self.confidence_interval = self.eval_epoch(phase="Testing")
        print(f"Testing accuracy : {self.testing_accuracy:.2f} +- {self.confidence_interval:.2f} %")

    def save_model(self, model_state=None, is_meta_training_phase=True):
        if model_state is None:
            model_state = self.best_state

        target_folder = os.path.join("checkpoint", self.args.dataset)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        if is_meta_training_phase:
            file_name = "{time}_{accuracy}_{interval}_{shot}_{grid_size}_{patch_ratio}_{epoch}_{best_epoch}_{lr}_{gamma}_{random}.pth".format(
                time=datetime.now().strftime("%m%d%H%M"),
                accuracy="{:.2f}".format(self.testing_accuracy),
                interval="{:.2f}".format(self.confidence_interval),
                shot=self.args.shot,
                grid_size=self.args.grid_size,
                patch_ratio="{:.1f}".format(self.args.patch_ratio),
                epoch=self.args.max_epoch,
                best_epoch=self.epoch_of_best_state,
                lr="{:.4f}".format(self.args.learning_rate),
                gamma="{:.1f}".format(self.args.gamma),
                random=self.args.random_seed)
        else:
            file_name = "{time}_{accuracy}_{epoch}_{best_epoch}_{bs}_{lr}_{gamma}.pth".format(
                time=datetime.now().strftime("%m%d%H%M"),
                accuracy="{:.2f}".format(self.testing_accuracy),
                epoch=self.args.max_epoch,
                best_epoch=self.epoch_of_best_state,
                bs=self.args.batch_size,
                lr="{:.4f}".format(self.args.learning_rate),
                gamma="{:.1f}".format(self.args.gamma))

        torch.save(model_state, os.path.join(target_folder, file_name))
