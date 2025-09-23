from torch.utils.data import Dataset
import pandas as pd

class HousingDataset(Dataset):
    def __init__(self, file_path:str, features:list[str], target_column:str=None):
        super().__init__()
        print("Reading csv files")
        self.df = pd.read_csv(file_path)
        self.df = self.df.fillna(0)
        self.target_column = target_column
        self.features = features
        print("Finished reading data")

        self.X = self.df[features].to_numpy()
        if target_column != None:
            self.Y = self.df[target_column].to_numpy()

        
    def __getitem__(self, index):
        if self.target_column==None:
            return self.X[index]
        else: 
            return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.df)
    
    
# if __name__ == "__main__":
#     from torch.utils.data import DataLoader
#     d = {'a':3, 'b':5}
#     print(len(d))

#     features=['LotFrontage']
#     target_column='SalePrice'
#     train_dataset = HousingDataset(
#         file_path="data/train.csv",
#         features=features,
#         target_column=target_column
#     )
#     test_dataset = HousingDataset(
#         file_path="data/test.csv",
#         features=features,
#     )
#     train_loader = DataLoader(dataset=train_dataset, batch_size=15, shuffle=True)
#     test_loader = DataLoader(dataset=test_dataset, batch_size=5, shuffle=False)
#     print(len(train_dataset))
#     print(len(test_dataset))
#     for x,y in train_loader:
#         print(x, y)

#     for x in test_loader:
#         print(x,y)




