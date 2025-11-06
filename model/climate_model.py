import torch
import torch.nn as nn

class TempPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(TempPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Helper functions
def save_model(model, path='model/trained_model.pth'):
    torch.save(model.state_dict(), path)

def load_model(input_size, path='model/trained_model.pth'):
    model = TempPredictor(input_size)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
