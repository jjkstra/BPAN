import torchvision.transforms as transforms
from utils import GaussianBlur

NORMALIZATION = {"mean": [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0],
                 "std": [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]}


def transform_manager(mode, args):
    if args.crop_mode == 'none':
        if mode == 'train':
            crop = transforms.RandomResizedCrop(84)
        else:
            crop = transforms.Compose([
                transforms.Resize([96, 96]),
                transforms.CenterCrop(84)
            ])
    elif args.crop_mode == 'grid':
        crop = transforms.Resize([84, 84])
    elif args.crop_mode == 'random':
        if mode == 'train':
            crop = transforms.RandomResizedCrop(84, args.crop_scale_in_train)
        else:
            crop = transforms.RandomResizedCrop(84, args.crop_scale_in_eval)
    else:
        raise ValueError('Error crop mode')

    flip_and_color_jitter = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4),
    ])

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**NORMALIZATION)
    ])

    return transforms.Compose([
        crop,
        flip_and_color_jitter,
        # GaussianBlur(),
        normalize
    ])
