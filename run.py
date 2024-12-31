import argparse
from modules import PreTrainer, MetaTrainer
from models import Classifier, BLAN
from setup import setup_random


def setup_argument_parser():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--data_root', type=str, default='../datasets')
    parser.add_argument('--dataset', type=str, default='mini_imagenet',
                        choices=['mini_imagenet', 'tiered_imagenet', 'fc100', 'cifar_fs', 'cub'])
    parser.add_argument('--pretrain_dir', type=str)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--emb_size', type=int, default=640)
    # models parameters
    parser.add_argument('--is_pretrained', action='store_true')
    parser.add_argument('--fine_tuning', action='store_true')
    parser.add_argument('--crop_mode', default='grid', choices=['none', 'grid', 'random'])
    # random
    parser.add_argument('--crop_scale_in_train', type=tuple, default=(0.08, 1.00))
    parser.add_argument('--crop_scale_in_eval', type=tuple, default=(0.08, 1.00))
    # grid
    parser.add_argument('--grid_size', type=int, default=3)
    parser.add_argument('--n_patch_views', type=int, default=10)
    parser.add_argument('--patch_ratio', type=float, default=1.5)
    # metric
    parser.add_argument('--metric', type=str, default='cos', choices=['l2', 'cos'])
    parser.add_argument('--tau', type=float, default=0.1)
    # training schedules
    parser.add_argument('--max_epoch', type=int, default=40,
                        help="""Maximum number of epochs for meta fine tuning. """)
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help="""Learning rate for meta fine-tuning outer loop. """)
    parser.add_argument('--gamma', type=float, default=0.2,
                        help="""learning rate decay ratio. """)
    parser.add_argument('--milestones', nargs='+', default=[20, 30], type=int,
                        help="""Where to decay learning rate. """)
    parser.add_argument('--momentum', type=float, default=0.9,
                        help="""Momentum for meta fine-tuning outer loop. """)
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help="""Weight decay for meta fine-tuning outer loop. """)
    # few-shot eval
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=5)
    parser.add_argument('--train_episode', type=int, default=100,
                        help="""Number of episodes used for training. """)
    parser.add_argument('--val_episode', type=int, default=500,
                        help="""Number of episodes used for validation. """)
    parser.add_argument('--test_episode', type=int, default=5000,
                        help="""Number of episodes used for testing. """)
    # SFC
    parser.add_argument('-sfc_lr', type=float, default=0.01, help='learning rate of SFC')
    parser.add_argument('-sfc_wd', type=float, default=0, help='weight decay for SFC weight')
    parser.add_argument('-sfc_update_step', type=float, default=100, help='number of updating step of SFC')
    parser.add_argument('-sfc_bs', type=int, default=4, help='batch size for finetune sfc')
    # running env
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--n_worker', type=int, default=4)

    return parser.parse_args()


if __name__ == '__main__':
    args = setup_argument_parser()
    print(args)
    setup_random(args.random_seed)

    if args.is_pretrained:
        model = Classifier(args)
        trainer = PreTrainer(args, model)
    else:
        model = BLAN(args)
        trainer = MetaTrainer(args, model)

    trainer.run()
