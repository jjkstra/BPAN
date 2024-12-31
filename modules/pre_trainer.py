from modules import Trainer
from copy import deepcopy


class PreTrainer(Trainer):
    def __init__(self, args, model):
        super().__init__(args, model)

    def run(self):
        min_loss = float('inf')
        min_loss_state = self.model.state_dict()

        for epoch in range(1, self.args.max_epoch + 1):
            print(f"Epoch {epoch}")
            self.timer.start()
            average_loss = self.train_epoch()
            validation_accuracy, _ = self.eval_epoch(phase="Validation")

            if average_loss < min_loss:
                min_loss = average_loss
                min_loss_state = deepcopy(self.model.state_dict())
                print("Ding ding ding! We found a min loss models!")

            self.compare_to_best_state(validation_accuracy, epoch)
            self.scheduler.step()

            epoch_time = self.timer.stop()
            print(f'roughly {(self.args.max_epoch - epoch) / 3600. * epoch_time:.2f} h left\n')

        self.test_model()
        self.save_model(is_meta_training_phase=False)
