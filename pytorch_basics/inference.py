import torch
from torch.utils.data import DataLoader
from dataset import HousingDataset


# Run inference on test dataset
features=['LotFrontage']
dataset = HousingDataset(
    file_path="data/test.csv",
    features=features,
)
data_size = len(dataset)
# loader = DataLoader(dataset, batch_size=100, shuffle=True)


model_save_path = "my_model.pt"
model = torch.load(model_save_path)

y_prediction = model(dataset.X) # [B, 1]

