#main.py
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from model import CNN, save_model
from dataloader import load_data
from training import back_pass, test
from visualization import  plot_mse, plot_classification_error, plot_all_weights

# Wybór urządzenia (GPU lub CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Inicjalizacja modelu, strat i optymalizatora
model = CNN().to(device)
batch_size = 128
train_loader, test_loader = load_data(batch_size=batch_size)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Parametry treningu
epochs = 60
best_loss = float('inf')  # Najlepsza Test Loss do tej pory

# Do monitorowania wyników
train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

for epoch in range(epochs):
    print(f"Epoch: {epoch + 1}/{epochs}")

    # Trening i testowanie
    train_loss, train_accuracy = back_pass(model, train_loader, loss_fn, optimizer, device)
    test_loss, test_accuracy = test(model, test_loader, loss_fn, device)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # Zapis modelu po każdej epoce
    save_model(model, path=f"model_epoch_{epoch + 1}.pth")

    # Aktualizacja najlepszej wartości Test Loss
    if test_loss < best_loss:
        best_loss = test_loss
        save_model(model, path="best_model.pth")  # Zapis najlepszego modelu

    # Aktualizacja współczynnika uczenia
    scheduler.step()


# Wykres MSE
plot_mse(train_losses, test_losses)

# Wykres błędu klasyfikacji
plot_classification_error(train_accuracies, test_accuracies)

# Wykresy wag
plot_all_weights(model)
