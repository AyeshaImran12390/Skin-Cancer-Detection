import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import gradio as gr
from fastapi import FastAPI
import uvicorn

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
            nn.Linear(32*16*16,64),
            nn.ReLU(),
            nn.Linear(64,2)
        )
    def forward(self,x):
        x = self.conv(x)
        x = self.fc(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

# Load weights
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

def predict_skin(image):
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output,1)
        return "Melanoma" if pred.item() == 0 else "Non-Melanoma"

# Gradio UI
iface = gr.Interface(
    fn=predict_skin,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Skin Cancer Detection",
    description="Upload a skin image"
)

# FastAPI App
app = FastAPI()

# Mount Gradio CORRECTLY
gr.mount_gradio_app(app, iface, path="/gradio")

@app.get("/")
def root():
    return {"message": "Skin Cancer Detection API is running!"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)
