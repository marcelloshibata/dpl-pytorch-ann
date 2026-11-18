# %%
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Equivalente ao random state 42
torch.manual_seed(42)

# Pipeline que realiza transformações nas imagens, padronizando-as.
tfm = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    # pixel_normalizado = (pixel - média) / desvio
    # Normalizamos assim para se adequar ao modelo de ImageNet.
])

train_ds = datasets.ImageFolder("data/train", transform=tfm) # X, y
print(train_ds.classes, train_ds.class_to_idx) # 'NORMAL': 0, 'PNEUMONIA': 1

# Gerador de Mini batches: Batches são par de images e labels.
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)

# Cria o iterador e pega next() pega o primero batch
images, labels = next(iter(train_dl))

# Images é um tensor com 32 imagens empilhadas, e labels é um tensor com 32 inteiros (0 = normal; 1 = Pneumonia)
print(images.shape, labels.shape, images.dtype, labels[:10])
# Resulta em torch.Tensor([32, 3, 224, 224])
# 32 (tamanho do batch (batch_size) -> 32 imagens processadas juntas)
# 3 Equivale aos canais de cores -> ImageNet espera 3 canais de cores enquanto Raio-X possuem só 1.
# 224, 224 altura e largura da imagem, os pixels.

# Em ML, fariamos model.fit(X, y), mas em DL o treino é feito da seguinte forma:
# 1. Pega 32 imagens (batches)
# 2. Calcula a loss nesses 32.
# 3. Atualiza os pesos.
# 4. Repete com os próximos 32.

# Uma passada por todos os batches, é uma época (epoch). Com 5216 imagens e batch 32, uma época tem 163 batches (len(train_dl))
print(len(train_dl)) # 163
print()

print(images.min(), images.max(), images.mean())
print(labels.bincount()) # quantos de cada classe(0 e 1) tem nesse batch?
# Resulta em 7 imagens normal e 25 de pneumonia, cerca de 78%. O que confirma que o dataset está desbalanceado.
# Um modelo ingênuo pode só chutar pneumonia que em 78% dos casos ele vai acertar, o que da uma loss baixa.
# Por isso mais pra frente temos que olhar outras métricas: recall, precisão e depois tratar o desbalanceamento.

test_ds = datasets.ImageFolder("data/test", transform=tfm)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)
# Para teste e validação, o shuffle deve ser falso. Isso porque não importa a ordem
# o modelo está simplesmente sendo avaliado.
# O shuffle no treino evita que o modelo veja os batches sempre na mesma ordem.
# Sem ele (no treino) o modelo pode aprender a ordem e isso prejudica a capacidade de predição.

val_ds = datasets.ImageFolder("data/val", transform=tfm)
val_dl = DataLoader(val_ds, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PneumoniaCNN(nn.Module):
    def __init__(self):
        super().__init__() # obrigatório: inicializa a lógica interna do nn.Module
        self.features = nn.Sequential(
            nn.Sequential(
                # 16 equivale aos canais, como max_depth da árvore, poucos dão pouca análise, muitos podem causar
                # overfitting.
                # de 3 canais, vai pra 16.
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ),
            # Chega aqui o tensor como [32, 16, 112, 112]
            nn.Sequential(
                # Os canais vão de 16 -> 32 para aumentar a capacidade de análise já que MaxPool2d
                # Diminui o tamanho pela metade
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ),
            # Chega aqui como tensor [32, 32, 56, 56]
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            # Sai dos blocos como tensor [32, 64, 28, 28]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Numero dado pela fórmula C * H * W (do tensor [B, C, H, W] -> [32, 64, 28, 28])
            nn.Linear(50176, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Criando uma instância da classe que contem a CNN
model = PneumoniaCNN().to(device)

# Fazendo x receber o batch [32, 3, 224, 224] que vem do DataLoader
x = images.to(device)

# Chama so o nn.Sequential() que passa por todas as camadas
print(model.features(x).shape) # Printar o formato do tensor após passar pelas camadas

print(model(x).shape)


print(sum(p.numel() for p in model.parameters()))
# - model.parameters() devolve todos os tensores de pesos e biases (de cada Conv2d e da Linear).
# - p.numel() é o número de elementos de cada tensor.
# - sum(...) soma tudo, e o resultado é a quantidade total de números que o treino vai ajustar.

# Em ML clássico se chama model.fit(X, y) e a biblioteca faz tudo. 
# No PyTorch se escreve esse "fit" na mão, e esse é o loop de treino. A ideia é simples: 
# o modelo tenta, erra, e corrige um pouquinho. Isso se repete muitas vezes.

# Uma analogia: você está aprendendo a acertar dardos num alvo.

# 1. Joga o dardo (o modelo faz uma predição).
# 2. Vê o quão longe do centro caiu (a loss mede o erro).
# 3. Descobre para onde ajustar o braço (o gradiente).
# 4. Ajusta o braço (o optimizer atualiza os pesos).
# 5. Joga de novo.

# forma de medir o erro
criterion = nn.BCEWithLogitsLoss()

# como corrigir os pesos
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# y = labels (0 normal 1 pneumonia) passados com .to(device) 
# para ficarem no mesmo dispositivo (cuda GPU)
y = labels.float().unsqueeze(1).to(device)

for epoch in range(3):
    model.train()
    running_loss = 0.0
    for images, labels in train_dl:
        # É necessário recalcular o x e y dentro do loop
        x = images.to(device)
        y = labels.float().unsqueeze(1).to(device)
        
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_ds)
    print(f"Epoch {epoch+1}: loss {epoch_loss:.4f}")


# Salvando os pesos do modelo.
# model.state_dict() é um dicionário com todos os pesos e biases (nome da camada -> tensor)
# salvamos só os números e não a classe.
# torch.save() grava esse dicionário em disco.
torch.save(model.state_dict(), "modelo_pneumonia.pt")
print("Pesos salvos")

# Para carregar depois:
model = PneumoniaCNN().to(device)
model.load_state_dict(torch.load("modelo_pneumonia.pt"))

# AVALIAÇÃO NA BASE DE TESTES

# Modo avaliação
model.eval()

all_probs = []
all_labels = []

with torch.no_grad(): # desliga cálculo de gradientes para economizar memória e tempo já que não vamos treinar
    for images, labels in test_dl:
        x = images.to(device)
        logits = model(x)

        # Aqui precisamos transformar o logit em probabilidade utilizando sigmoid
        probs = torch.sigmoid(logits).squeeze(1).cpu() # probabilidade de pneumonia
        # .cpu necessário pois traz o resultado pra memória da CPU. sklearn não trabalha com GPU        

        all_probs.append(probs)
        all_labels.append(labels)

probs = torch.cat(all_probs) # junta a lista de tensores em um tensor só
labels = torch.cat(all_labels) # junta a lista de tensores em um tensor só
preds = (probs > 0.5).long() # predições: 1 se probabilidade > 0.5, senão 0

print(confusion_matrix(labels, preds))
print(classification_report(labels, preds, target_names=["NORMAL", "PNEUMONIA"]))
print("AUC: ", roc_auc_score(labels, probs))