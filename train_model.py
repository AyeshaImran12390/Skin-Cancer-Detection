# skin_cancer_cnn.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

# Dataset paths
train_path = 'C:/Users/HP/Downloads/DEEP PROJECT/DATA/New folder'  
test_path  = 'C:/Users/HP/Downloads/DEEP PROJECT/DATA/New folder'  

train_data = datasets.ImageFolder(root='C:/Users/HP/Downloads/DEEP PROJECT/DATA/New folder', transform=transform)
test_data  = datasets.ImageFolder(root='C:/Users/HP/Downloads/DEEP PROJECT/DATA/New folder' , transform=transform)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=16, shuffle=False)

# CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*16*16, 64),
            nn.ReLU(),
            nn.Linear(64,2)
        )
    def forward(self,x):
        x = self.conv(x)
        x = self.fc(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Testing Accuracy
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Test Accuracy:", round(correct/total*100,2), "%")

# Save the trained model weights
# torch.save(model.state_dict(), r"C:\Users\HP\Downloads\DEEP PROJECT\DATA\model_weights.pth")
# print("Model weights saved successfully at 'model_weights.pth'")

# OR, if you prefer to save the full model (architecture + weights)
torch.save(model, r"C:\Users\HP\Downloads\DEEP PROJECT\DATA\model.pth")
print("Full model saved successfully at 'model.pth'")

torch.save(model.state_dict(), "model.pth")
print("MODEL SAVED AS model.pth")

