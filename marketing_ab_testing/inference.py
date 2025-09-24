import torch
from torch.utils.data import random_split, DataLoader
from datasets import Marketing_Dataset


feature_list=['total ads','most ads hour']
full_dataset = Marketing_Dataset(file_path="marketing_ab_testing/data/marketing_AB.csv",
                            feature_list=feature_list,
                            target_column='converted')
dataset_size = len(full_dataset)
generator2 = torch.Generator().manual_seed(42)
train_df, validate_df, test_df = random_split(full_dataset,[0.7,0.15,0.15],generator=generator2)
test_loader = DataLoader(test_df,batch_size=len(test_df),shuffle=False)

model_save_path = "marketing_ab_testing/my_model.pt"
model = torch.load(model_save_path)
loss_binary = torch.nn.BCEWithLogitsLoss()

for x, y in test_loader:
    y_prediction = model(x)
    y = y.float()
    error = loss_binary(y_prediction,y)

########################################

threshold = 0.1
print(f'\n threshold = {threshold}')
y_prediction = torch.functional.F.sigmoid(y_prediction) #convert to values between 0-1
y_prediction = torch.where(y_prediction < threshold,0,1)

y_prediction_minus_y = y_prediction - y
y_prediction_plus_y = y_prediction + y

# Count true positive
pos_pred = int(torch.sum(torch.eq(y_prediction,1)))
print(f'\n --- total positive prediction: {pos_pred}')
true_pos = int(torch.sum(torch.eq(y_prediction_plus_y,2)))
false_pos = int(torch.sum(torch.eq(y_prediction_minus_y,1)))
false_neg = int(torch.sum(torch.eq(y_prediction_minus_y,-1)))
true_neg = int(torch.sum(torch.eq(y_prediction_plus_y,0)))

# Calculate Precision, Recall, Accuracy
precision = true_pos / (true_pos+false_pos)
recall = true_pos / (true_pos+false_neg)
accuracy = (true_pos+true_neg) / (true_pos+true_neg+false_neg+false_pos)
print(f'>>>> precision: {precision}')
print(f'>>>> recall:  {recall}')
print(f'>>>> accuracy: {accuracy}')
print('_____________')
