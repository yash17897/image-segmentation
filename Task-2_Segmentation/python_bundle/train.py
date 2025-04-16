
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CityscapesDataset
from metrics import iou_score, dice_score, pixel_accuracy
import torchvision.transforms as transforms
from model import UNet  # Assumes model.py is provided

def train():
    transform = transforms.Compose([transforms.Resize((256, 512)), transforms.ToTensor()])
    train_dataset = CityscapesDataset('data/train/images', 'data/train/masks', transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    model = UNet(num_classes=20).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        model.train()
        for images, masks in train_loader:
            images, masks = images.cuda(), masks.cuda()
            outputs = model(images)
            loss = criterion(outputs, masks.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

        torch.save(model.state_dict(), f"unet_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()
