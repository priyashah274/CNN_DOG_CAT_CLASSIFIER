import torch, os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from utils.transforms import train_transform, test_transform
from utils.data_setup import extract_and_prepare_data
from model.cnn import CNNClassifier
from config import *
from tqdm.auto import tqdm

def train_step(model, dataloader, loss_fn, optimizer):
    model.train()
    train_loss, train_acc = 0, 0
    for X, y in dataloader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        train_acc += (y_pred.argmax(1) == y).sum().item() / len(y_pred)
    return train_loss/len(dataloader), train_acc/len(dataloader)

def test_step(model, dataloader, loss_fn):
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            y_pred = model(X)
            test_loss += loss_fn(y_pred, y).item()
            test_acc += (y_pred.argmax(1) == y).sum().item() / len(y_pred)
    return test_loss/len(dataloader), test_acc/len(dataloader)

def train(model, train_loader, test_loader, optimizer, loss_fn, epochs):
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_loader, loss_fn, optimizer)
        test_loss, test_acc = test_step(model, test_loader, loss_fn)
        print(f"Epoch {epoch+1}: TLoss={train_loss:.4f}, TAcc={train_acc:.4f}, ValLoss={test_loss:.4f}, ValAcc={test_acc:.4f}")
        results["train_loss"].append(train_loss); results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss); results["test_acc"].append(test_acc)
    return results

if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    extract_and_prepare_data(TRAIN_ZIP_PATH, TEST_ZIP_PATH, TRAIN_DIR, TEST_DIR)

    train_data = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
    test_data = datasets.ImageFolder(root=TEST_DIR, transform=test_transform)
    class_names = train_data.classes

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = CNNClassifier().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    from timeit import default_timer as timer
    start = timer()
    results = train(model, train_loader, test_loader, optimizer, loss_fn, NUM_EPOCHS)
    end = timer()
    print(f"Training time: {end-start:.2f} sec")

    from utils.plot_utils import plot_loss_curves
    plot_loss_curves(results)

    torch.save(model.state_dict(), "cnn_model.pth")
