import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import load_data
from models.cnn import CNN
from models.autoencoder import AutoEncoder



def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

def evaluate(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def train():
    trainloader, testloader = load_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    autoencoder = AutoEncoder()
    autoencoder.load_state_dict(torch.load('../models/autoencoder.py'))
    autoencoder.eval()  # Mode évaluation
    autoencoder.to(device)
    # Modèle
    model = CNN().to(device)
    initialize_weights(model)

    # Optimiseur et Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Fonction de perte
    loss_fn = nn.CrossEntropyLoss()

    # TensorBoard
    writer = SummaryWriter()

    epochs = 50
    best_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Reconstruire les images avec l'Auto-Encoder
            with torch.no_grad():
                inputs = autoencoder(inputs)

            # Entraîner le modèle CNN
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()


            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(trainloader)
        accuracy = 100 * correct / total

        # Scheduler
        scheduler.step()

        # Évaluer sur le dataset de test
        test_accuracy = evaluate(model, testloader, device)

        # Sauvegarde du meilleur modèle
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), 'best_mnist_model.pth')
            print(f"Modèle sauvegardé avec une précision de {best_accuracy:.2f}% sur le test")

        # TensorBoard
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Accuracy/train", accuracy, epoch)
        writer.add_scalar("Accuracy/test", test_accuracy, epoch)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")

    writer.close()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True  # Accélération GPU
    train()
