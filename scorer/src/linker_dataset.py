import os
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

class LinkerDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

def default_collate(data_list):
    return Batch.from_data_list(data_list)