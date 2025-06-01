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
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred,y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model, dataloader, loss_fn):
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model, train_loader, test_loader, optimizer, loss_fn, epochs):
    results = {"train_loss": [],"train_acc": [],"test_loss": [],"test_acc": []}

    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn)

        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # 6. Return the filled results at the end of the epochs
    return results

if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    extract_and_prepare_data(TRAIN_ZIP_PATH, TEST_ZIP_PATH, TRAIN_DIR, TEST_DIR)
    print("Extracted test and train files")
    train_data = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform, target_transform=None)
    test_data = datasets.ImageFolder(root=TEST_DIR, transform=test_transform)
    class_names = train_data.classes
    
    print("Creating Dataloader")
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print("Creating Model...")
    model = CNNClassifier().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    print("Training model")
    from timeit import default_timer as timer
    start = timer()
    results = train(model, train_loader, test_loader, optimizer, loss_fn, NUM_EPOCHS)
    end = timer()
    print(f"Training time: {end-start:.2f} sec")

    from utils.plot_utils import plot_loss_curves
    plot_loss_curves(results)

    torch.save(model.state_dict(), "cnn_model.pth")
