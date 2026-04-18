import os
import torch
import const
from torch.utils.data import Dataset
from torch_geometric.data import Batch

## Linker
class LinkerDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

def default_collate(data_list):
    return Batch.from_data_list(data_list)

## PROTAC
class PROTACDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

def collate(batch):
    out = {}

    for i, data in enumerate(batch):
        for key, value in data.items():
            out.setdefault(key, []).append(value)

    for key, value in out.items():
        if key in const.DATA_LIST_ATTRS:
            continue
        if key in const.DATA_ATTRS_TO_PAD:
            out[key] = torch.nn.utils.rnn.pad_sequence(value, batch_first=True, padding_value=0)
            continue
        raise Exception(f'Unknown batch key: {key}')

    atom_mask = (out['fragment_mask'].bool() | out['linker_mask'].bool()).to(const.TORCH_INT)
    out['atom_mask'] = atom_mask[:, :, None]

    batch_size, n_nodes = atom_mask.size()

    edge_mask = atom_mask[:, None, :] * atom_mask[:, :, None]
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=const.TORCH_INT).unsqueeze(0)
    edge_mask *= diag_mask
    out['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)
    # out['edge_mask'] = edge_mask
    

    for key in const.DATA_ATTRS_TO_ADD_LAST_DIM:
        if key in out.keys():
            out[key] = out[key][:, :, None]

    return out


def create_template(tensor, fragment_size, linker_size, fill=0):
    values_to_keep = tensor[:fragment_size]
    values_to_add = torch.ones(linker_size, tensor.shape[1], dtype=values_to_keep.dtype)
    values_to_add = values_to_add * fill
    return torch.cat([values_to_keep, values_to_add], dim=0)


def create_templates_for_linker_generation(data, linker_sizes):
    """
    Takes data batch and new linker size and returns data batch where fragment-related data is the same
    but linker-related data is replaced with zero templates with new linker sizes
    """
    decoupled_data = []
    for i, linker_size in enumerate(linker_sizes):
        data_dict = {}
        fragment_mask = data['fragment_mask'][i].squeeze()
        fragment_size = fragment_mask.sum().int()
        for k, v in data.items():
            if k == 'num_atoms':
                # Computing new number of atoms (fragment_size + linker_size)
                data_dict[k] = fragment_size + linker_size
                continue
            if k in const.DATA_LIST_ATTRS:
                # These attributes are written without modification
                data_dict[k] = v[i]
                continue
            if k in const.DATA_ATTRS_TO_PAD:
                # Should write fragment-related data + (zeros x linker_size)
                fill_value = 1 if k == 'linker_mask' else 0
                template = create_template(v[i], fragment_size, linker_size, fill=fill_value)
                if k in const.DATA_ATTRS_TO_ADD_LAST_DIM:
                    template = template.squeeze(-1)
                data_dict[k] = template

        decoupled_data.append(data_dict)

    return collate(decoupled_data)
