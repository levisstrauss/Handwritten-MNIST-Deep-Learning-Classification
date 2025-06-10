import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt


class Trainer:
    """
    Professional training class with comprehensive monitoring.
    """

    def __init__(self, model, device, criterion, optimizer, scheduler=None):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()

            if batch_idx % 100 == 0:
                current_acc = 100. * correct_predictions / total_samples
                pbar.set_postfix({
                    'Loss': f'{running_loss / (batch_idx + 1):.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct_predictions / total_samples

        return epoch_loss, epoch_acc

    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_samples += targets.size(0)
                correct_predictions += (predicted == targets).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100. * correct_predictions / total_samples

        return val_loss, val_acc

    def train(self, train_loader, val_loader, epochs, save_dir='models/'):
        """Complete training loop."""
        best_val_acc = 0.0

        print("Starting training...")
        print("=" * 50)

        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validation
            val_loss, val_acc = self.validate(val_loader)

            # Update scheduler
            if self.scheduler:
                if hasattr(self.scheduler, 'step'):
                    self.scheduler.step()

            # Save metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, f'{save_dir}best_model.pth')

            print(f'Epoch [{epoch + 1}/{epochs}] - '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        print("=" * 50)
        print(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")

        return self.train_losses, self.train_accuracies, self.val_losses, self.val_accuracies

    def plot_training_history(self):
        """Plot training metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss plot
        ax1.plot(self.train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy plot
        ax2.plot(self.train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(self.val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Training and Validation Accuracy', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()