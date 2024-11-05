import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableActivation(nn.Module):
    def __init__(self, in_features):
        super(LearnableActivation, self).__init__()

        self.linear = nn.Linear(in_features, in_features)

    def forward(self, x):

        return torch.sigmoid(self.linear(x))

class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):


        super().__init__()

        self.linear_or_not = True  # default is linear agent
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear agent
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer agent
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.activations = nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
                self.activations.append(LearnableActivation(hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        if self.linear_or_not:
            # If linear agent
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = self.activations[layer](self.linears[layer](h))
                #h = F.relu(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)

class LearnableActor(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):

        super().__init__()

        self.linear_or_not = True  # default is linear agent
        self.num_layers = num_layers

        self.activative = torch.tanh

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear agent
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer agent
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()

            self.activations = nn.ModuleList([LearnableActivation(hidden_dim) for _ in range(num_layers - 1)])

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
                self.activations.append(LearnableActivation(hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        if self.linear_or_not:
            # If linear agent
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = self.activations[layer](self.linears[layer](h))
                #h = self.activative((self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)

class LearnableCritic(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):

        super().__init__()

        self.linear_or_not = True  # default is linear agent
        self.num_layers = num_layers

        self.activative = torch.tanh

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear agent
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer agent
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            #self.activations = nn.ModuleList()
            self.activations = nn.ModuleList([LearnableActivation(hidden_dim) for _ in range(num_layers - 1)])

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
                self.activations.append(LearnableActivation(hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        if self.linear_or_not:
            # If linear agent
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = self.activations[layer](self.linears[layer](h))
                #h = self.activative((self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)
