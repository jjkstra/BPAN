import torch
from modules import Trainer
from utils import strip_prefix


class MetaTrainer(Trainer):
    def __init__(self, args, model):
        super().__init__(args, model)

    def load_model(self):
        checkpoint_path = self.args.pretrain_dir
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        backbone_missing_keys, _ = self.model.backbone.load_state_dict(
            strip_prefix(state_dict, 'backbone.'), strict=False)
        if len(backbone_missing_keys) > 0:
            raise ValueError(f"Missing keys for backbone: {backbone_missing_keys}")

    def run(self):
        if self.args.fine_tuning:
            self.load_model()
        for epoch in range(1, self.args.max_epoch + 1):
            print(f"Epoch {epoch}")
            self.timer.start()
            self.train_epoch()
            validation_accuracy, _ = self.eval_epoch(phase="Validation")
            self.compare_to_best_state(validation_accuracy, epoch)
            self.scheduler.step()
            epoch_time = self.timer.stop()
            print(f'roughly {(self.args.max_epoch - epoch) / 3600. * epoch_time:.2f} h left\n')

        self.test_model()
        self.save_model()
