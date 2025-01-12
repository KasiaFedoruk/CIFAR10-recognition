#model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Warstwy konwolucyjne
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.batch_norm4 = nn.BatchNorm2d(512)

        # Automatyczne obliczanie wymiaru wejściowego do fc1
        dummy_input = torch.zeros(1, 3, 32, 32)
        with torch.no_grad():
            dummy_output = self.pool(
                self.batch_norm4(
                    F.relu(self.conv5(
                        self.pool(
                            self.batch_norm3(
                                F.relu(self.conv4(
                                    self.pool(
                                        self.batch_norm2(
                                            F.relu(self.conv3(
                                                self.pool(
                                                    self.batch_norm1(
                                                        F.relu(self.conv2(
                                                            self.pool(F.relu(self.conv1(dummy_input)))
                                                        ))
                                                    ))
                                                ))
                                            ))
                                        ))
                                    ))
                                ))
                            ))
            self.fc_input_dim = dummy_output.numel()

        # Warstwy w pełni połączone
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.layer_norm1 = nn.LayerNorm(512)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.layer_norm2 = nn.LayerNorm(256)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(self.batch_norm1(F.relu(self.conv2(x))))
        x = self.pool(self.batch_norm2(F.relu(self.conv3(x))))
        x = self.pool(self.batch_norm3(F.relu(self.conv4(x))))
        x = self.pool(self.batch_norm4(F.relu(self.conv5(x))))

        # Dopasowanie do fc1
        x = x.view(-1, self.fc_input_dim)

        # Warstwy w pełni połączone z LayerNorm i Dropout
        x = F.relu(self.layer_norm1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.layer_norm2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def save_model(model, path="model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model zapisano w {path}")

def load_model(model, path="model.pth", device=torch.device("cpu")):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model załadowano z {path}")
    return model
