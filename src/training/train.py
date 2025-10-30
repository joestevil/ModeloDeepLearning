"""
Esqueleto de entrenamiento (PyTorch) — completar según modelo y dataset.
"""
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    # TODO: completar: dataset, transforms, modelo, hiperparámetros
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Usando device:', device)

    # ejemplo placeholder
    model = nn.Sequential(nn.Flatten(), nn.Linear(224*224*3, 10))
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # loader placeholder
    train_loader = []  # reemplazar con DataLoader real

    # bucle de entrenamiento (placeholder)
    for epoch in range(1, 11):
        print(f'Epoch {epoch}')
        # average_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # print('Loss:', average_loss)


if __name__ == '__main__':
    main()
