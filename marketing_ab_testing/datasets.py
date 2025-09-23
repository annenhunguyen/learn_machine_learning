from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch

class Marketing_Dataset(Dataset):
    def __init__(self,file_path:str,feature_list:list[str],target_column:str=None):
        super().__init__()
        self.df = pd.read_csv(file_path)
        self.features = self.df[feature_list].to_numpy()
        self.target = self.df[target_column].to_numpy().reshape(-1,1)
        print(self.features.shape)
        print(self.target.shape)

    def __getitem__(self, index):
        feature_batch = torch.tensor(self.features[index], dtype=torch.float32)
        target_batch = torch.tensor(self.target[index], dtype=torch.float32)
        return feature_batch, target_batch
    
    def __len__(self):
        return len(self.df)