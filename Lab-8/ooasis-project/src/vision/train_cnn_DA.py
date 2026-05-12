from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from src.vision.image_dataset import EuroSATDataset
from torch import nn
from src.vision.cnn_model import SimpleCNN
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

TRAIN_DIR = Path("data/processed/images/train")
TEST_DIR = Path("data/processed/images/test")
BATCH_SIZE = 16
EPOCHS = 20 
LEARNING_RATE = 0.001
MODEL_PATH = Path("models/cnn_model.pt")
CLASS_NAMES_PATH = Path("models/cnn_classes.txt")

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def create_dataloaders():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = EuroSATDataset(
        root_dir=TRAIN_DIR,
        transform=train_transform
    )

    test_dataset = EuroSATDataset(
        root_dir=TEST_DIR,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("=== DataLoader Inspection (with Augmentation) ===")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    print(f"Classes: {train_dataset.class_names}")

    return train_loader, test_loader, train_dataset.class_names

def train_model(model, train_loader, device):
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()

    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {average_loss:.4f}")

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print("\n=== Evaluation Results ===")
    print(f"Accuracy with Augmentation: {accuracy:.4f}")
    return accuracy

def plot_confusion_matrix(model, test_loader, class_names, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    
    # Crear la figura explícitamente para evitar problemas de renderizado
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=ax)
    plt.title("Confusion Matrix (Data Augmentation)")
    
    # Nombre de archivo solicitado con _DA
    save_path = Path("reports/confusion_matrix_DA.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Guardar antes de plt.show() para asegurar que no se limpie el buffer
    plt.savefig(save_path, bbox_inches="tight")
    plt.close() # Cerrar para liberar memoria
    print(f"Confusion matrix saved at: {save_path}")

def save_model(model, class_names):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    with open(CLASS_NAMES_PATH, "w") as f:
        for name in class_names:
            f.write(name + "\n")
    print("\n=== Model Saved Successfully ===")

def main():
    train_loader, test_loader, class_names = create_dataloaders()
    device = get_device()
    print(f"Using device: {device}")

    model = SimpleCNN(num_classes=len(class_names)).to(device)
    train_model(model, train_loader, device)
    evaluate_model(model, test_loader, device)
    plot_confusion_matrix(model, test_loader, class_names, device)
    save_model(model, class_names)

if __name__ == "__main__":
    main()