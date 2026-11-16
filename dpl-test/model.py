# %%
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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