import torch
from models.cnn import CNN
from utils.dataset import load_data

def test():
    _, testloader = load_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CNN().to(device)
    model.load_state_dict(torch.load('best_mnist_model.pth', map_location=device))
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

    print(f"Précision sur le dataset de test : {100 * correct / total:.2f}%")

if __name__ == "__main__":
    test()
