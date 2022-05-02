import torch
from torch.utils.data import Dataset
from utils_CBRS import *


class OnlineContinuum(Dataset):
    def __init__(self, dataset, classes_per_task=2, iid_continuum=False, iid_task=False, transform=None):
        super(OnlineContinuum).__init__()
        self.transform = transform
        self.inputs, self.labels = torch.tensor([]), torch.tensor([])
        self.load_dataset(dataset)
        self.configure_data_order(classes_per_task, iid_continuum, iid_task)

    def __getitem__(self, index):
        if self.transform is None:
            return self.inputs[index], self.labels[index], index
        else:
            return self.transform(self.inputs[index]), self.labels[index], index

    def __len__(self):
        return self.labels.shape[0]

    def load_dataset(self, name):
        self.inputs, self.labels = load_data(name, train = True)
        self.inputs1, self.labels1 = load_data(name, train = False)

    def configure_data_order(self, classes_per_task, iid_continuum, iid_task):
        # If continuum is iid just shuffle it
        if iid_continuum:
            order = torch.randperm(len(self))
            self.inputs, self.labels = self.inputs[order], self.labels[order]
            return

        # Shuffle each class in-place (argsort is deterministic)
        classes, counts = torch.unique(self.labels, return_counts=True)
        order = torch.argsort(self.labels)
        start = 0
        for i, count in enumerate(counts):
            end = count + start
            order[start:end] = order[start:end][torch.randperm(end - start)]
            start = end

        # Configure the order and shuffle the samples of each task if requested
        n_tasks = int(np.ceil(classes.shape[0] / classes_per_task))
        task_borders = torch.zeros(n_tasks + 1, dtype=torch.int)
        for task in range(n_tasks):
            a, b = task * classes_per_task, min((task + 1) * classes_per_task, counts.shape[0])
            task_borders[task+1] = torch.sum(counts[a:b]) + task_borders[task]
            if iid_task:
                start, end = task_borders[task], task_borders[task+1]
                order[start:end] = order[start:end][torch.randperm(int(end - start))]
        self.inputs, self.labels = self.inputs[order], self.labels[order]

    def reduce_size(self, retain_percentage):
        size = len(self)
        indexes = torch.sort(torch.randperm(size)[:int(size * retain_percentage)])[0]
        self.inputs, self.labels = self.inputs[indexes], self.labels[indexes]
        
    def get_data(self):
        return self.inputs, self.labels

    def create_imbalances(self, max_imbalance, steps):
        exponents = torch.linspace(-max_imbalance, 0, steps)
        imbalance_ratios = torch.tensor(10.0).pow(exponents).tolist()
        classes, counts = torch.unique(self.labels, return_counts=True)
        copy = int(np.ceil(classes.shape[0] / len(imbalance_ratios)))
        imbalance_ratios = torch.tensor(imbalance_ratios * copy)
        indexes = torch.randperm(int(imbalance_ratios.shape[0]))[:int(classes.shape[0])]
        imbalance_ratios = imbalance_ratios[indexes]

        indexes = []
        for index, c in enumerate(self.labels):
            if torch.rand([]) < imbalance_ratios[c]:
                indexes.append(index)

        indexes = torch.tensor(indexes)
        self.inputs, self.labels = self.inputs[indexes], self.labels[indexes]
