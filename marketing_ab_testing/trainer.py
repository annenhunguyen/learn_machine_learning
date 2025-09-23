import torch 
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split

######################################
# ------ Datasets -----------

from datasets import Marketing_Dataset

feature_list=['total ads','most ads hour']
full_dataset = Marketing_Dataset(file_path="marketing_ab_testing/data/marketing_AB.csv",
                            feature_list=feature_list,
                            target_column='converted')
dataset_size = len(full_dataset)
# train_split = int(0.7*dataset_size)
# test_split = int(0.15*dataset_size)
generator2 = torch.Generator().manual_seed(42)
train_df, validate_df, test_df = random_split(full_dataset,[0.7,0.15,0.15],generator=generator2)

train_loader = DataLoader(train_df, batch_size=1024, shuffle=True)
validate_loader = DataLoader(validate_df, batch_size=128, shuffle=False)


######################################
# ------ Model -----------

from model import MyModel

num_features = len(feature_list)
num_outputs = {'5unit':5,
               '1unit':1}
model = MyModel(num_features=num_features,num_outputs=num_outputs)
optimizer = SGD(model.parameters(), lr=1e-3, momentum=0.1, dampening=0.1, weight_decay=0.1)


######################################
# ------ Trainer -----------

num_epochs = 2
loss_binary = torch.nn.BCEWithLogitsLoss()
for e in range(num_epochs):
    iteration = 0
    previous_val_loss = float('inf')
    validation_tolerance = 0
    for x, y in train_loader:
        y_hat = model(x)
        y = y.float()   #converting boolean to 1 and 0
        loss = loss_binary(y_hat,y)
        if iteration%500 == 0: 
            print(f'\n >>>> LOSS: {loss}, iteration: {iteration}')

        # print("\n----- Gradients (before backward()):")
        # for name, param in model.named_parameters():
        #     print(f"{name}.data: {param.data}", "\n",f'grad: {param.grad}')

        optimizer.zero_grad()
        loss.backward()

        # print("\n--- After backward() and before optimizer.step() ---")
        # print("Parameters:")
        # for name, param in model.named_parameters():
        #     print(f"{name}.data: {param.data}", "\n",f'grad: {param.grad}')

        optimizer.step() # update the weights using the mentioned Learning rate and gradient computed above
        
        # print("\n ---- Parameters (after update):")
        # for name, param in model.named_parameters():
        #     # The parameters have been updated based on the gradients
        #     print(f"{name}.data: {param.data}", "\n",f'grad: {param.grad}')


        # ---Validation---------
        if iteration%500 == 0:
            sum_val_loss = 0
            for x_val, y_val in validate_loader:
                y_hat_val = model(x_val)
                y_val = y_val.float()
                loss_val_onebatch = loss_binary(y_hat_val,y_val)
                sum_val_loss = sum_val_loss + loss_val_onebatch
            loss_val = sum_val_loss/len(validate_loader)
        
            if loss_val > previous_val_loss:
                validation_tolerance += 1
            previous_val_loss = loss_val
            print(f'VAL LOSS: {loss_val}, validation tolerance: {validation_tolerance}')

        if validation_tolerance > 100:
            break

        iteration += 1
        # print('----------------------------------')

model_save_path = "marketing_ab_testing/my_model.pt"
torch.save(model, model_save_path)