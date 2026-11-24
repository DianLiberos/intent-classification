from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, activation_function='ReLU', dropout_p=0.5):
        super().__init__()
        if activation_function == 'ReLU':
            act = nn.ReLU
        elif activation_function == 'SiLU':
            act = nn.SiLU
        else:
            raise ValueError(f"Unknown activation_function: {activation_function}")

        self.sequential_stack = nn.Sequential(
            nn.Linear(2000, 512),
            act(),
            nn.Dropout(dropout_p),
            nn.Linear(512, 128),
            act(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 5)
        )
    def forward(self, x):
        logits = self.sequential_stack(x)
        return logits

