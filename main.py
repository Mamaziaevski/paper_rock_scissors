from fastapi import FastAPI, HTTPException, UploadFile, File
import uvicorn
import torch
import io
import torch.nn as nn
from torchvision import transforms
from PIL import Image


class Game(nn.Module):
    def __init__(self):
        super().__init__()

        self.first = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # nn.Conv2d(256, 512, kernel_size=3, padding=1)
        )

        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 5)
        )

    def forward(self, games):
        games = self.first(games)
        games = self.second(games)
        return games

transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
])

app = FastAPI()
model = Game()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

mapped = {
    0: 'paper',
    1: 'scissor',
    2: 'rock',
}

@app.post("/predict")
async def check_image(image: UploadFile = File(...)):
    try:
        image_file = await image.read()
        if not image_file:
            raise HTTPException(status_code=404, detail="File not found")

        img = Image.open(io.BytesIO(image_file)).convert("RGB")
        img_tensor = transforms(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            pred = output.argmax(dim=1).item()
        return{"result": mapped[pred]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__name__":
    uvicorn.run(app, host="127.0.0.1", port=8000)