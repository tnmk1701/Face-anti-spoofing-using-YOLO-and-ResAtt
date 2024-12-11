import torch
import torch.nn as nn
import numpy as np
# import torchvision.models as models
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm  # Import tqdm for the progress bar


class BasicBlockWitOuthAttention(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlockWitOuthAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
      #  self.attention = SelfAttention(out_channels)

    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.functional.relu(out)
      #  out = self.attention(out)
        return out

class ASDataset(Dataset):
    def __init__(self, client_file: str, imposter_file: str, transforms=None):
        with open(client_file, "r") as f:
            client_files = f.read().splitlines()
        with open(imposter_file, "r") as f:
            imposter_files = f.read().splitlines()
        self.labels = torch.cat((torch.ones(len(client_files)), torch.zeros(len(imposter_files))))
        self.imgs = client_files + imposter_files
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img = Image.open(img_name)
        label = self.labels[idx]
        if self.transforms:
            img = self.transforms(img)
        return img, label


# Transformations for the input images
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ResNet18 model with SelfAttention
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class BasicBlockWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlockWithAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.attention = SelfAttention(out_channels)

    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.functional.relu(out)
        out = self.attention(out)
        return out

class ResNet18WithAttention(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNet18WithAttention, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = nn.Sequential(
            BasicBlockWitOuthAttention(64, 64),
            BasicBlockWitOuthAttention(64, 64)
        )
        self.attention1 = SelfAttention(64)
        self.layer2 = nn.Sequential(
            BasicBlockWitOuthAttention(64, 128, stride=2),
            BasicBlockWitOuthAttention(128, 128)
        )
        self.attention2 = SelfAttention(128)
        self.layer3 = nn.Sequential(
            BasicBlockWitOuthAttention(128, 256, stride=2),
            BasicBlockWitOuthAttention(256, 256)
        )
        self.attention3 = SelfAttention(256)
        self.layer4 = nn.Sequential(
            BasicBlockWitOuthAttention(256, 512, stride=2),
            BasicBlockWitOuthAttention(512, 512)
        )
        self.attention4 = SelfAttention(512)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.attention1(x)
        x = self.layer2(x)
        x = self.attention2(x)
        x = self.layer3(x)
        x = self.attention3(x)
        x = self.layer4(x)
        x = self.attention4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

ResNet18WithAttention()