import torch
from torch.optim import SGD
from torch.utils.data import random_split, DataLoader

from dataset import HousingDataset
from model import MyModel

features=['LotFrontage']
target_column='SalePrice'
dataset = HousingDataset(
    file_path="data/train.csv",
    features=features,
    target_column=target_column
)
data_size = len(dataset)
train_split = int(0.7*data_size)
test_split = int(0.15*data_size)
train_dataset, val_dataset, test_dataset = random_split(dataset, [
                                                train_split, 
                                                data_size - train_split - test_split,
                                                test_split
])

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=50, shuffle=False)

n_epochs = 1
learning_rate = 0.001
model = MyModel(num_features=len(features), num_output=1)
optimizer = SGD(model.parameters(), lr=learning_rate, weight_decay=0.1, momentum=0.1, dampening=0.1)


global_step = 0
for e in range(n_epochs):
    for x, y in train_loader: # x[B, #features] y[B,1]
        global_step+=1
        y_hat = model(x) # [B, 1]
        loss = ((y-y_hat)**2).mean() #[B,1]

        # 2. Before `backward()`: Gradients are None
        # The .grad attribute is not yet populated
        print("---Gradients (before backward()):")
        for name, param in model.named_parameters():
            print(f"{name}.data: {param.data} , grad: {param.grad}")

        optimizer.zero_grad()
        loss.backward() # computes the gradient for all parameters in the path of the loss calculation from input
        print("\n--- After backward() and before optimizer.step() ---")
        print("Parameters:")
        for name, param in model.named_parameters():
            print(f"{name}.data: {param.data} , grad: {param.grad}")

        optimizer.step() # update the weights using the mentioned Learning rate and gradient computed above
        print(">>>>>Parameters (after update):")
        for name, param in model.named_parameters():
            # The parameters have been updated based on the gradients
            print(f"{name}.data: {param.data}, grad: {param.grad}")

model_save_path = "my_model.pt"
torch.save(model, model_save_path)
