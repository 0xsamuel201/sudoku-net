import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import the lightweight model from mobilenet.py
from mobilenet import MiniMobileNet

def parse_args():
    """
    Parse command line arguments for training configuration.
    """
    parser = argparse.ArgumentParser(description="Train MiniMobileNet on MNIST Dataset")
    
    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Input batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
    
    # Output Files
    parser.add_argument('--weight-file', type=str, default='minimobile_mnist.pth', 
                        help='Filename to save the best model weights')
    parser.add_argument('--plot-file', type=str, default='mnist_training_results.png', 
                        help='Filename to save the training visualization plot')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # ---------------------------------------------------------
    # 1. Environment Setup
    # ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")

    # ---------------------------------------------------------
    # 2. Data Preparation (MNIST)
    # ---------------------------------------------------------
    # Note: We do NOT use Resize(224) here because MiniMobileNet 
    # is optimized to process the native 28x28 MNIST images.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Global Mean and Std for MNIST
    ])

    print("[*] Loading MNIST dataset...")
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, 
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, 
                                             shuffle=False, num_workers=2)

    # ---------------------------------------------------------
    # 3. Model, Loss, and Optimizer Initialization
    # ---------------------------------------------------------
    # num_classes=10 (digits 0-9), in_channels=1 (Grayscale)
    model = MiniMobileNet(num_classes=10, in_channels=1).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Tracking metrics across epochs
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    best_acc = 0.0 

    # ---------------------------------------------------------
    # 4. Training Loop
    # ---------------------------------------------------------
    for epoch in range(args.epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar info
            progress_bar.set_postfix(loss=running_loss/len(trainloader), 
                                     acc=100.*correct/total)

        train_losses.append(running_loss / len(trainloader))
        train_accs.append(100. * correct / total)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        epoch_val_loss = val_loss / len(testloader)
        epoch_val_acc = 100. * val_correct / val_total
        test_losses.append(epoch_val_loss)
        test_accs.append(epoch_val_acc)
                
        print(f"[*] Summary Epoch {epoch+1}: Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

        # Save the best model based on validation accuracy
        if epoch_val_acc > best_acc:
            print(f"    [+] New best model found! Saving weights to {args.weight_file}")
            torch.save(model.state_dict(), args.weight_file)
            best_acc = epoch_val_acc

    print("\n[!] Training complete.")

    # ---------------------------------------------------------
    # 5. Visualization and Export
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, args.epochs+1), train_losses, label='Train Loss', color='blue', marker='o')
    plt.plot(range(1, args.epochs+1), test_losses, label='Test Loss', color='red', marker='x')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, args.epochs+1), train_accs, label='Train Acc', color='blue', marker='o')
    plt.plot(range(1, args.epochs+1), test_accs, label='Test Acc', color='green', marker='x')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(args.plot_file, dpi=300)
    print(f"[*] Plot saved to {args.plot_file}")

if __name__ == "__main__":
    main()