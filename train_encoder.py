import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from models.autoencoder import AutoEncoder

# Charger les données MNIST
def load_data(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),  # Aucune normalisation spécifique
    ])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader

# Fonction d'évaluation
def evaluate(model, testloader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, _ in testloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, inputs)
            total_loss += loss.item()
    return total_loss / len(testloader)

# Sauvegarde des reconstructions
def save_reconstruction_image(inputs, outputs, epoch):
    inputs = inputs[:8]  # Prends les 8 premières images
    outputs = outputs[:8]
    images = torch.cat((inputs, outputs), dim=0)  # Concatène
    save_image(images, f'./reconstructions_epoch_{epoch}.png', nrow=8)

# Fonction d'entraînement
def train_autoencoder(epochs=50, batch_size=128, learning_rate=0.001):
    # Charger les données
    trainloader, testloader = load_data(batch_size)

    # Initialiser le modèle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoEncoder().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    loss_fn = nn.MSELoss()  # Optionnel : remplacez par BCEWithLogitsLoss si nécessaire

    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, _ in trainloader:
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, inputs)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(trainloader)
        test_loss = evaluate(model, testloader, loss_fn, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Test Loss: {test_loss:.4f}")

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), './autoencoder.pth')
            print("Modèle sauvegardé avec un Test Loss de {:.4f}".format(best_loss))

        # Sauvegarder les reconstructions
        if epoch % 5 == 0:
            with torch.no_grad():
                sample_inputs, _ = next(iter(testloader))
                sample_inputs = sample_inputs.to(device)
                sample_outputs = model(sample_inputs)
                save_reconstruction_image(sample_inputs, sample_outputs, epoch)

if __name__ == "__main__":
    train_autoencoder()
