import torch
import torch.functional as F

class MyModel(torch.nn.Module):
    def __init__(self, num_features:int, num_output:int):
        super().__init__() # <--- Parent class constructor
        self.linear_layer = torch.nn.Linear(in_features=num_features, out_features=num_output, bias=True, dtype=torch.float64) # just contains the weight
        self.relu_layer = torch.nn.ReLU()
        self.relu_layer = torch.nn.Sigmoid()
        self.linear_layer2 = torch.nn.Linear(in_features=num_output, out_features=1, bias=True, dtype=torch.float64) # just contains the weight

        self.model_part1 = torch.nn.Sequential([
            self.linear_layer,
            self.relu_layer,
            self.linear_layer2,
            self.relu_layer
        ])

        self.model_part2 = torch.nn.Sequential([
            self.linear_layer,
            self.relu_layer,
            self.linear_layer2,
            self.relu_layer
        ])
    
    def forward(self, x):
        y1 = self.model_part1(x)
        y=self.model_part2(y1)
        return y


# if __name__=="__main__":
#     class ABC:
#         def __init__(self):
#             self.x=10
#         def multiply_five(self):
#             self.x = self.x*5
#         def __call__(self):
#             return self.x

#     a = ABC()
#     a.multiply_five()
#     print(a())
