import torch
import numpy as np
from torch.utils.data import Sampler


class TaskSampler(Sampler):

    def __init__(self, labels, n_task, n_way, n_shot, n_query):
        super().__init__(data_source=None)
        self.n_task = n_task  # the number of iterations in the dataloader
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query

        labels = np.array(labels)  # all data label
        self.m_ind = []  # the data index of each class
        for i in range(max(labels) + 1):
            ind = np.argwhere(labels == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_task

    def __iter__(self):
        for i_batch in range(self.n_task):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_way]  # random sample num_class indexes,e.g. 5
            for c in classes:
                l = self.m_ind[c]  # all data indexes of this class
                pos = torch.randperm(len(l))[:self.n_shot + self.n_query]  # sample n_per data index of this class
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd
            yield batch

    def episodic_collate_fn(self, input_data):
        true_class_ids = list({x[1] for x in input_data})
        all_images = torch.cat([x[0].unsqueeze(0) for x in input_data])
        query_labels = torch.arange(self.n_way, dtype=torch.long).repeat(self.n_query)
        return all_images, query_labels
