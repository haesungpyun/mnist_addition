import pickle
import torch
from torch.utils.data import DataLoader, Dataset


class MNISTDataset(Dataset):
    def __init__(self, dataset, test:bool = False):
        super().__init__()
        self.dataset = dataset
        
        
        if test:
            self.num1 = dataset[0][:, None, ]
            self.num2 = dataset[0][:, None, ]
            self.labels = dataset[1]
        
        else:
            num_pairs = dataset[0]
            self.labels = dataset[1]
            self.num1 = num_pairs[:, 0, None, ]
            self.num2 = num_pairs[:, 1, None, ]
            assert len(self.num1) == len(self.num2)
                     
        assert len(self.num1) == len(self.labels)
        
    def __len__(self,):
        return len(self.num1)
    
    def __getitem__(self, idx):
        return self.num1[idx].float(), self.num2[idx].float(), self.labels[idx]
    

def get_loader(batch_size):
    """Data For Train and Validation"""
    with open('../data/MNIST_pair/training_tuple.pkl', 'rb') as f:
        data = pickle.load(f)
    
    """Split Train and Valdiation set"""
    tr_len = int(data[0].size(0) * 0.75)

    perm_idxs = torch.randperm(data[0].size(0))

    train_idx = perm_idxs[:tr_len]
    valid_idx = perm_idxs[tr_len: ]

    train_data = data[0][train_idx]
    train_label = data[1][train_idx]
    assert len(train_data) == len(train_label)

    valid_data = data[0][valid_idx]
    valid_label = data[1][valid_idx]
    assert len(valid_data) == len(valid_label)
        
    """Normalize"""
    mean, std = train_data.float().mean(), train_data.float().std()
    train_data = ((train_data - mean) / std, train_label)
    valid_data = ((valid_data - mean) / std , valid_label)
    
#     train_data = (train_data, train_label)
#     valid_data = (valid_data, valid_label)

    """Data For Test"""
    with open('../data/MNIST_pair/test.pkl', 'rb') as f:
        test_data = pickle.load(f)
    test_label = test_data[1]
    
    """Normalize"""
    # mean, std = test_data[0].float().mean(), test_data[0].float().std()
    test_data = ((test_data[0] - mean) / std, test_label)

#     test_data = (test_data[0], test_label)

    
    """Make Dataset"""
    train_dataset = MNISTDataset(train_data, test=False)
    val_dataset = MNISTDataset(valid_data, test=False)
    test_dataset = MNISTDataset(test_data, test=True)
    
    """Make DataLoader"""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, val_loader, test_loader