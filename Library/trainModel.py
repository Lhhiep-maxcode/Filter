import zipfile
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import xml.etree.ElementTree as ET
import torch
from torchvision.transforms import v2
import albumentations as A
from torch.utils.data import DataLoader, Dataset
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torchvision.models as models


def train_model(model, optimizer, criterion, num_epochs=35):
    best_test_loss = 1e9
    train_loss_history = []
    test_loss_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader)
        for images, landmarks in train_pbar:
            images = images.to(device)
            landmarks = landmarks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, landmarks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            train_pbar.set_postfix({"Train loss": loss.item()})

        train_loss = running_loss / len(train_loader.dataset)
        train_loss_history.append(train_loss)

        # Validation (optional)
        model.eval()
        test_loss = 0.0
        test_pbar = tqdm(test_loader)
        with torch.no_grad():
            for images, landmarks in test_pbar:
                images = images.to(device)
                landmarks = landmarks.to(device)

                outputs = model(images)
                loss = criterion(outputs, landmarks)

                test_loss += loss.item() * images.size(0)
                test_pbar.set_postfix({"Test loss": loss.item()})

        test_loss = test_loss / len(test_loader.dataset)
        test_loss_history.append(test_loss)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

        # Save the best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print('Model improved, saving...')
            
    plt.plot(np.arange(len(train_loss_history)), np.array(train_loss_history), color='red', label='Train')
    plt.plot(np.arange(len(test_loss_history)), np.array(test_loss_history), color='green', label='Test')
    plt.legend()
    plt.show()
    
    return model