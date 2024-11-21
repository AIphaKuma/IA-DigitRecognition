import torch
from torchvision import datasets, transforms
from models.autoencoder import AutoEncoder
import matplotlib.pyplot as plt

# Charger le modèle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
autoencoder = AutoEncoder().to(device)
autoencoder.load_state_dict(torch.load('./autoencoder.pth', map_location=device))
autoencoder.eval()

# Charger une image de MNIST
transform = transforms.Compose([
    transforms.ToTensor(),  # Pas de normalisation ici
])
dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# Tester une image
inputs, _ = next(iter(dataloader))
inputs = inputs.to(device)

# Passer dans l'AutoEncoder
with torch.no_grad():
    outputs = autoencoder(inputs)

# Si les données sont entre [-1, 1], les ramener à [0, 1]
inputs = inputs.cpu().numpy()
outputs = outputs.cpu().numpy()

# Afficher l'entrée et la reconstruction
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Input")
plt.imshow(inputs[0][0], cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Reconstructed")
plt.imshow(outputs[0][0], cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
