import matplotlib.pyplot as plt
import numpy as np

def plot_mse(train_losses, test_losses):
    """
    Rysuje wykres błędu średniokwadratowego (MSE) dla zbioru treningowego i testowego.
    """
    train_mse = np.square(train_losses)  # Przekształcenie strat na MSE
    test_mse = np.square(test_losses)
    plt.figure()
    plt.plot(train_mse, label='Train MSE')
    plt.plot(test_mse, label='Test MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('Mean Squared Error (MSE) over Epochs')
    plt.show()

def plot_classification_error(train_accuracies, test_accuracies):
    """
    Rysuje wykres błędu klasyfikacji (1 - Accuracy) dla zbioru treningowego i testowego.
    """
    train_errors = [100 - acc for acc in train_accuracies]  # Błąd klasyfikacji
    test_errors = [100 - acc for acc in test_accuracies]
    plt.figure()
    plt.plot(train_errors, label='Train Classification Error')
    plt.plot(test_errors, label='Test Classification Error')
    plt.xlabel('Epoch')
    plt.ylabel('Classification Error (%)')
    plt.legend()
    plt.title('Classification Error over Epochs')
    plt.show()

def plot_weights(model, layer_name="fc1"):
    """
    Rysuje histogram wag dla wybranej warstwy w pełni połączonej.
    """
    weights = None
    for name, param in model.named_parameters():
        if layer_name in name and "weight" in name:
            weights = param.data.cpu().numpy().flatten()
            break

    if weights is not None:
        plt.figure()
        plt.hist(weights, bins=50, alpha=0.75)
        plt.title(f"Histogram wag dla warstwy: {layer_name}")
        plt.xlabel("Wartości wag")
        plt.ylabel("Liczba wystąpień")
        plt.show()
    else:
        print(f"Warstwa {layer_name} nie została znaleziona.")

def plot_all_weights(model):
    """
    Rysuje histogramy wag dla wszystkich warstw w pełni połączonych.
    """
    layers = [name for name, param in model.named_parameters() if "weight" in name]
    for layer_name in layers:
        plot_weights(model, layer_name)
