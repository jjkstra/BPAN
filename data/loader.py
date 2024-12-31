from torch.utils.data import DataLoader
from data import DatasetGenerator, TaskSampler


def get_dataloader(mode, args):
    dataset = DatasetGenerator(mode, args)

    if mode == 'train':
        if args.is_pretrained:
            return DataLoader(dataset=dataset,
                              batch_size=args.batch_size,
                              num_workers=args.n_worker,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True)
        episode = args.train_episode
    elif mode == 'val':
        episode = args.val_episode
    elif mode == 'test':
        episode = args.test_episode
    else:
        raise ValueError('Unknown mode')

    sampler = TaskSampler(dataset.get_labels(), episode, args.way, args.shot, args.query)

    return DataLoader(dataset=dataset,
                      batch_sampler=sampler,
                      num_workers=args.n_worker,
                      pin_memory=True,
                      collate_fn=sampler.episodic_collate_fn)
