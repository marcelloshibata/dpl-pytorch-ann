# %%
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import os
import re
from sklearn.model_selection import StratifiedGroupKFold
from torchvision import models
import matplotlib.pyplot as plt

# Equivalente ao random state 42
torch.manual_seed(26)

# Pipeline que realiza transformações nas imagens, padronizando-as.
tfm = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    # pixel_normalizado = (pixel - média) / desvio
    # Normalizamos assim para se adequar ao modelo de ImageNet.
])

train_tfm = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.75, 1.5)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

train_ds = datasets.ImageFolder("data/train", transform=tfm) # X, y
print(train_ds.classes, train_ds.class_to_idx) # 'NORMAL': 0, 'PNEUMONIA': 1

train_ds_aug = datasets.ImageFolder("data/train", transform=train_tfm)

# Gerador de Mini batches: Batches são par de images e labels.
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)

# Cria o iterador e pega next() pega o primero batch
images, labels = next(iter(train_dl))

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

# função ID do paciente a partir do nome do arquivo
def patient_id(path):
    # os.path.basename pega só o nome do arquivo, sem as pastas
    # ex.: "data/train\\PNEUMONIA\\person321_bacteria_1483.jpeg" -> "person321_bacteria_1483.jpeg"
    nome = os.path.basename(path)

    # Pneumonia: o nome começa com "person" + número do paciente
    # ex.: person321_bacteria_1483.jpeg -> paciente "person321"
    if nome.startswith("person"):
        # re.match procura o padrão no INÍCIO do texto; \d+ = um ou mais dígitos
        # .group(1) devolve só o que ficou dentro dos parênteses (ex.: "person321")
        return re.match(r"(person\d+)", nome).group(1)

    # Normal: o nome tem "IM-" + número do paciente
    # ex.: IM-0115-0001.jpeg ou NORMAL2-IM-0115-0001.jpeg -> paciente "IM-0115"
    # re.search procura o padrão em QUALQUER posição (por causa do prefixo "NORMAL2-")
    return re.search(r"(IM-\d+)", nome).group(1)

# train_ds.samples é a lista de (caminho, rótulo) de cada imagem, na ordem dos índices
# Cria uma lista com o paciente (grupo) de cada imagem. O "_" ignora o rótulo
groups = [patient_id(p) for p, _ in train_ds.samples]

# Rótulo (0 = NORMAL, 1 = PNEUMONIA) de cada imagem, criado pelo ImageFolder
targets = train_ds.targets

# Posição de cada imagem: [0, 1, 2, ..., 5215]. O split trabalha com essas posições
idx = list(range(len(train_ds)))

# Divisor que faz DUAS coisas ao mesmo tempo:
# - agrupado: todas as imagens de um paciente ficam do mesmo lado (sem vazamento)
# - estratificado: mantém a proporção NORMAL/PNEUMONIA nos dois lados
# n_splits=6 -> divide em 6 partes (~16,7% dos pacientes em cada)
# shuffle=True -> embaralha os pacientes antes de dividir
# random_state=42 -> resultado reproduzível
sgkf = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42)

# sgkf.split gera as 6 divisões possíveis; next() pega só a primeira
# Ela devolve os índices do treino e os índices da validação
train_idx, val_idx = next(sgkf.split(idx, targets, groups))

# ---- Verificação: nenhum paciente pode estar nos dois conjuntos ----
# Conjuntos (set) com os pacientes de cada lado
g_train = {groups[i] for i in train_idx}
g_val = {groups[i] for i in val_idx}
# "&" é a interseção: pacientes presentes nos dois. Tem que ser 0
print("pacientes em comum:", len(g_train & g_val))

# Rótulos como tensor, para poder indexar com a lista de posições
t = torch.tensor(targets)
# Tamanhos dos dois conjuntos (esperado: 4346 e 870)
print(len(train_idx), len(val_idx))
# t[train_idx] pega os rótulos só das imagens de treino; bincount conta quantos 0 e 1
# Serve para conferir se a proporção de classes se manteve (~25,7% NORMAL)
print(t[train_idx].bincount(), t[val_idx].bincount())

# # Subset do Pytorch, cria uma "visão" do dataset só com os índices escolhidos
# # sem copiar nada
train_sub = Subset(train_ds_aug, train_idx)
val_sub = Subset(train_ds, val_idx)

# Treino: shuffle=True (embaralha os batches a cada época)
train_dl = DataLoader(train_sub, batch_size=32, shuffle=True)
# Validação: shuffle=False (só avalia, a ordem não importa)
val_dl = DataLoader(val_sub, batch_size=32, shuffle=False)

# tamanhos
print(len(train_sub), len(val_sub)) # 4346 treino e 870 teste

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
counts = t[train_idx].bincount()
pos_weight = torch.tensor([counts[0].item() / counts[1].item()], device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# y = labels (0 normal 1 pneumonia) passados com .to(device) 
# para ficarem no mesmo dispositivo (cuda GPU)
y = labels.float().unsqueeze(1).to(device)

def evaluate(model, loader, criterion):
    # Modo avaliação
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_labels = []

    # Sem gradientes
    with torch.no_grad():
        for images, labels in loader:
            x = images.to(device)
            y = labels.float().unsqueeze(1).to(device)

            logits = model(x)
            loss = criterion(logits, y)
            running_loss += loss.item() * images.size(0)

            # .cpu necessário pois sklearn só trabalha com cpu
            # Aqui precisamos transformar o logit em probabilidade utilizando sigmoid
            probs = torch.sigmoid(logits).squeeze(1).cpu()

            # Junta a probabilidade e as labels nos seus respectivos conjuntos
            all_probs.append(probs)
            all_labels.append(labels)

    avg_loss = running_loss / len(loader.dataset)
    # torch.cat junta a lista de tensores em um tensor só
    return avg_loss, torch.cat(all_probs), torch.cat(all_labels)

# Cria arquitetura da ResNet18 e baixa os pesos treinados no ImageNet
# DEFAULT pede os melhores pesos disponiveis.
# Weights=none criaria a mesma arquitetura com pesos aleatórios
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# A resNet termina em uma camada chamada fc (fully connected), que
# originalmente tem 512 entradas e 1000 saidas
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(device)

x = images.to(device)        # um batch qualquer [32, 3, 224, 224]
print(model(x).shape)        # esperado: torch.Size([32, 1])
print(sum(p.numel() for p in model.parameters()))

# como corrigir os pesos
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_val_loss = float("inf")
train_losses, val_losses = [], []

num_epochs = 8

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_dl:
        x = images.to(device)
        y = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_dl.dataset)

    val_loss, val_probs, val_labels = evaluate(model, val_dl, criterion)
    val_auc = roc_auc_score(val_labels, val_probs)

    train_losses.append(epoch_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1} | treino {epoch_loss:.4f} | val {val_loss:.4f} | val AUC {val_auc:.3f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # Salvando os pesos do modelo.
        # model.state_dict() é um dicionário com todos os pesos e biases (nome da camada -> tensor)
        # salvamos só os números e não a classe.
        # torch.save() grava esse dicionário em disco.
        torch.save(model.state_dict(), "melhor_modelo_resnet.pt")
        print(" -> melhor modelo salvo")


# Para carregar depois (a ResNet, e não mais a PneumoniaCNN):
# recria a arquitetura IDÊNTICA à do treino, com a fc trocada ANTES do load_state_dict
# weights=None: não baixa o ImageNet, os pesos vêm do arquivo salvo
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load("melhor_modelo_resnet.pt"))
model = model.to(device)

# Versão antiga (CNN pequena), mantida só como referência:
# model = PneumoniaCNN().to(device)
# model.load_state_dict(torch.load("modelo_pneumonia.pt"))

# GRAD CAM
model.eval()

def gradcam(model, x):
    # x: UMA imagem [1, 3, 224, 224] com o mesmo transformer do teste
    model.eval()
    model.zero_grad()

    # Dicionários locais: guardam a ativação e o gradiente da última camada convolucional
    activations = {}
    gradients = {}

    def fwd_hook(module, inp, out):
        activations["a"] = out # [1, 512, 7, 7]
        out.register_hook(lambda g: gradients.__setitem__("g", g)) # captura o gradiente que volta

    # O hook só existe DENTRO desta função. Se ficasse registrado fora, ele dispararia
    # em qualquer forward (como na evaluate, que usa torch.no_grad) e daria erro
    handle = model.layer4.register_forward_hook(fwd_hook)
    try:
        with torch.enable_grad(): # força o cálculo de gradientes, mesmo se chamada dentro de no_grad
            logit = model(x)
            logit.backward() # gradiente do logit de pneumonia
    finally:
        handle.remove() # desliga o hook SEMPRE, mesmo se der erro

    a = activations["a"]
    g = gradients["g"]

    pesos = g.mean(dim=(2, 3), keepdim=True) # importancia de cada canal
    cam = (pesos * a).sum(dim=1, keepdim=True) # soma ponderada dos mapas
    cam = torch.relu(cam) # só o que empurra A FAVOR de pneumonia

    cam = torch.nn.functional.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8) # normaliza para 0..1

    return cam.squeeze().detach().cpu(), torch.sigmoid(logit).item()

# Função de plotar

mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def mostrar (ax, img_tensor, cam, titulo):
    img = (img_tensor.cpu() * std + mean).clamp(0, 1) # desfaz normalização
    ax.imshow(img.permute(1, 2, 0)) # [C, H, W] -> [H, W, C]
    ax.imshow(cam, cmap="jet", alpha=0.4) # mapa de calor semitransparente por cima
    ax.set_title(titulo, fontsize=8)
    ax.axis("off")

# AVALIAÇÃO NA BASE DE TESTES
test_ds = datasets.ImageFolder("data/test", transform=tfm)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)

model.load_state_dict(torch.load("melhor_modelo_resnet.pt"))  # ou o nome do último arquivo salvo
test_loss, test_probs, test_labels = evaluate(model, test_dl, criterion)

preds = (test_probs > 0.5).long()

grupos = {
    "Verdadeiro positivo":  (preds == 1) & (test_labels == 1),
    "Verdadeiro negativo":  (preds == 0) & (test_labels == 0),
    "Falso positivo":       (preds == 1) & (test_labels == 0),
    "Falso negativo":       (preds == 0) & (test_labels == 1),
}

g = torch.Generator().manual_seed(0)
n_cols = 4
fig, axes = plt.subplots(len(grupos), n_cols, figsize=(3 * n_cols, 3.2 * len(grupos)))

for r, (nome, mascara) in enumerate(grupos.items()):
    idxs = mascara.nonzero().squeeze(1)
    escolhidos = idxs[torch.randperm(len(idxs), generator=g)[:n_cols]]
    for c in range (n_cols):
        ax = axes[r][c]
        if c < len(escolhidos):
            i = escolhidos[c].item()
            img, _ = test_ds[i]
            cam, p = gradcam(model, img.unsqueeze(0).to(device))
            mostrar(ax, img, cam, f"{nome}\nprob pneumonia={p:.2f}")
        else:
            ax.axis("off")

plt.tight_layout()
plt.show()

print(roc_auc_score(test_labels, test_probs))
print(classification_report(test_labels, (test_probs > 0.5).long(), target_names=["NORMAL", "PNEUMONIA"]))

# print(confusion_matrix(labels, preds))
# print(classification_report(labels, preds, target_names=["NORMAL", "PNEUMONIA"]))
# print("AUC: ", roc_auc_score(labels, probs))