import matplotlib.pyplot as plt

def plot_loss_curves(results):
    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, results["train_loss"], label="Train Loss")
    plt.plot(epochs, results["test_loss"], label="Test Loss")
    plt.legend(); plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, results["train_acc"], label="Train Acc")
    plt.plot(epochs, results["test_acc"], label="Test Acc")
    plt.legend(); plt.title("Accuracy")
    plt.show()
