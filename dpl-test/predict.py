import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# O MESMO pré-processamento do teste
tfm = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def carregar_modelo(caminho_pesos):
    model = models.resnet18(weights=None)                          # só a arquitetura
    model.fc = nn.Linear(model.fc.in_features, 1)                  # fc ANTES do load
    model.load_state_dict(torch.load(caminho_pesos, map_location=device))
    return model.to(device).eval()

def prever(model, caminho_imagem, limiar=0.5):
    img = Image.open(caminho_imagem)                 # abre a imagem (PIL)
    x = tfm(img).unsqueeze(0).to(device)             # [3,224,224] -> [1,3,224,224]
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()        # probabilidade de pneumonia
    return prob, ("PNEUMONIA" if prob > limiar else "NORMAL")

model = carregar_modelo("melhor_modelo_resnet.pt")
prob, classe = prever(model, "caminho/da/radiografia.jpeg")
print(f"{classe} (prob. de pneumonia = {prob:.2f})")