import torch

class MyModel(torch.nn.Module):
    def __init__(self,num_features:int,num_outputs:dict):
        super().__init__()
        self.linear_5unit = torch.nn.Linear(in_features=num_features, out_features=num_outputs['5unit'], bias=True, dtype=torch.float32)
        self.linear_1unit = torch.nn.Linear(in_features=num_outputs['5unit'], out_features=num_outputs['1unit'], bias=True,dtype=torch.float32)
        self.relu_layer = torch.nn.ReLU()
        # self.sigmoid_layer = torch.nn.Sigmoid()

        self.model = torch.nn.Sequential(
            self.linear_5unit,
            self.relu_layer,
            self.linear_1unit,
        )

    def forward(self,features):
        y1 = self.model(features)
        return y1

