"""Train ResNet-18 on MNIST and print held-out test loss and accuracy."""

import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def main(epochs=5):
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "mnist"
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train, test = [
        DataLoader(
            datasets.MNIST(root, train=training, download=True, transform=transform),
            batch_size=128,
            shuffle=training,
        )
        for training in (True, False)
    ]
    model = models.resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(epochs):
        for images, labels in train:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(
                model(images.to(device)), labels.to(device)
            )
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs} complete ({device})", flush=True)
    model.eval()
    correct, test_loss = 0, 0.0
    with torch.inference_mode():
        for images, labels in test:
            logits = model(images.to(device))
            test_loss += nn.functional.cross_entropy(
                logits, labels.to(device), reduction="sum"
            ).item()
            correct += (logits.argmax(1).cpu() == labels).sum().item()
    print(f"Final test loss (cross-entropy): {test_loss / len(test.dataset):.6f}")
    print(f"Final test accuracy: {100 * correct / len(test.dataset):.2f}%")


if __name__ == "__main__":
    main()
