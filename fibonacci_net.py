import torch
import torch.nn as nn
import torch.optim as optim
from alembic.command import history
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from imblearn.over_sampling import RandomOverSampler


class Avg2MaxPooling(nn.Module):
    def __init__(self, kernel_size=3, stride=2, padding=1):
        """Novel Avg-2Max Pooling layer (as per paper)"""
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size, stride, padding)
        self.max_pool = nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, x):
        # Emphasize edges by subtracting twice the max pooled value from avg pooled value
        return self.avg_pool(x) - (self.max_pool(x) + self.max_pool(x))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution with ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=kernel_size // 2,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class FibonacciNet(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), num_classes=1):
        super().__init__()

        # Block 1 (21 filters)
        self.block1 = nn.Sequential(
            nn.Conv2d(input_shape[0], 21, kernel_size=3, padding=1),
            nn.BatchNorm2d(21),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # -> 21 x 112 x 112
        )

        # Block 2 (34 filters)
        self.block2 = nn.Sequential(
            nn.Conv2d(21, 34, kernel_size=3, padding=1),
            nn.BatchNorm2d(34)
        )
        self.block2_relu = nn.ReLU()  # Separate for skip connection
        self.block2_pool = nn.MaxPool2d(kernel_size=2)  # -> 34 x 56 x 56

        # Block 3 (55 filters)
        self.block3 = nn.Sequential(
            nn.Conv2d(34, 55, kernel_size=3, padding=1),
            nn.BatchNorm2d(55)
        )
        self.block3_relu = nn.ReLU()  # Separate for skip connection
        self.block3_pool = nn.MaxPool2d(kernel_size=2)  # -> 55 x 28 x 28

        # PCB1: Block 2 -> Block 4
        self.pcb1 = nn.Sequential(
            nn.Conv2d(34, 24, kernel_size=3, padding=1),    # 24 x 56 x 56
            Avg2MaxPooling(),       # 24 x 28 x 28
            nn.Conv2d(24, 24, kernel_size=3, padding=1),    # 24 x 28 x 28
            Avg2MaxPooling()        # 24 x 14 x 14
        )

        # Block 4 (89 filters)
        self.block4 = nn.Sequential(
            nn.Conv2d(55, 89, kernel_size=3, padding=1),
            nn.BatchNorm2d(89),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # -> 89 x 14 x 14
        )

        # PCB2: Block 3 -> Block 5
        self.pcb2 = nn.Sequential(
            nn.Conv2d(55, 24, kernel_size=3, padding=1),  # 24 x 28 x 28
            Avg2MaxPooling(),  # 24 x 14 x 14
            nn.Conv2d(24, 24, kernel_size=3, padding=1),  # 24 x 14 x 14
            Avg2MaxPooling()  # 24 x 7 x 7
        )

        # Block 5 (144 filters)
        self.block5 = nn.Sequential(
            nn.Conv2d(89 + 24, 144, kernel_size=3, padding=1),
            nn.BatchNorm2d(144),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # -> 144 x 7 x 7
        )

        # Block 6 (233 filters, DWSC)
        self.block6 = DepthwiseSeparableConv(144 + 24, 233)     # 233 x 7 x 7

        # Block 7 (377 filters, DWSC)
        self.block7 = DepthwiseSeparableConv(233, 377)      # 377 x 7 x 7

        # Output
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(377, num_classes)
        self.sigmoid = nn.Sigmoid()

    def backward(self, x):
        # Block 1
        x = self.block1(x)

        # Block 2
        x = self.block2(x)
        x2 = self.block2_relu(x)    # Save for pcb1
        x = self.block2_pool(x2)

        # Block 3
        x = self.block3(x)
        x3 = self.block3_relu(x)    # Save for pcb1
        x = self.block3_pool(x)

        # Block 4
        x = self.block4(x)

        # PCB1 processing
        pcb1 = self.pcb1(x2)     # Using x2 from block 2
        # 24 x 14 x 14

        # Concat block4 and pcb1
        x = torch.cat([x, pcb1], dim=1)     # (89 + 24) x 14 x 14

        # Block 5
        x = self.block5(x)      # 144 x 7 x 7

        # PCB2 processing
        pcb2 = self.pcb2(x3)    # Using x3 from block 3
        # 24 x 7 x 7

        # Concat Block 5 and pcb2
        x = torch.cat([x, pcb2], dim=1)     # (144 + 24) x 7 x 7

        # Block 6 & 7
        x = self.block6(x)      # 233 x 7 x 7
        x = self.block7(x)      # 377 x 7 x 7

        #Output
        x = self.global_pool(x)     #(377,1,1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x

class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        super().__init__()
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        image = Image.open(img_path).convert('RGB')
        label = float(self.dataframe.iloc[idx]['category_encoded'])

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

class FibonacciNetTrainer:
    def __init__(self, model, device, batch_size=16, learning_rate=1e-4):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            outputs = outputs.view(-1)      # Flatten for BCE loss
            loss = self.criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                outputs = outputs.view(-1)          # Flatten for BCE loss
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def predict(self, test_loader):
        self.model.eval()
        predictions = []
        probabilities = []
        true_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)

                outputs = self.model(inputs)
                outputs = outputs.view(-1)      #Flatten

                probs = outputs.cpu().numpy()
                preds = (probs > 0.5).astype(int)

                probabilities.extend(probs)
                predictions.extend(preds)
                true_labels.extend(labels.cpu().numpy())

        return np.array(true_labels), np.array(predictions), np.array(probabilities)

    def train(self, train_loader, val_loader, epochs=5):
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

        for epoch in range (epochs):
            #train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc = self.validate(val_loader)

            # Store history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        return history

def plot_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=300)
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()

def plot_roc_curve(y_true, y_probs):
    fpr, tpr, threshold = roc_curve(y_true=y_true, y_score=y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (Area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - ROC')
    plt.legend(loc="lower right")
    plt.savefig("roc_auc.png", dpi=300)
    plt.show()

    return roc_auc

def process_data(df_path):
    #Load data
    df = pd.read_csv(df_path)

    #convert labels to numeric
    label_encoder = LabelEncoder()
    df['category_encoded'] = label_encoder.fit_transform(df['label'])
    df = df[['image_path', 'category_encoded']]

    # Handle class imbalance with oversampling
    ros = RandomOverSampler(random_state=42)
    x_resampled, y_resampled = ros.fit_resample(df[['image_path']], df['category_encoded'])

    # Create new dataframe with balanced classes
    df_resampled = pd.DataFrame(x_resampled, columns=['image_path'])
    df_resampled['category_encoded'] = y_resampled
    df_resampled['category_encoded'] = df_resampled['category_encoded'].astype(str)

    # Split the data
    train_df, temp_df = train_test_split(
        df_resampled,
        train_size=0.8,
        shuffle=True,
        random_state=42,
        stratify=df_resampled['category_encoded']
    )

    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        shuffle=True,
        random_state=42,
        stratify=temp_df['category_encoded']
    )

    return train_df, valid_df, test_df

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Process data
    train_df, valid_df, test_df = process_data("brain_tumor_mri.csv")

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = CustomImageDataset(train_df, transform=transform)
    valid_dataset = CustomImageDataset(valid_df, transform=transform)
    test_dataset = CustomImageDataset(test_df, transform=transform)

    # Create dataloaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model
    model = FibonacciNet(input_shape=(3, 224, 224), num_classes=1)

    # Train model
    trainer = FibonacciNetTrainer(model, device, batch_size=batch_size)
    history = trainer.train(train_loader, valid_loader, epochs=5)

    # Plot training history
    plot_history(history)

    # Evaluate on test set
    y_true, y_pred, y_probs = trainer.predict(test_loader)

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))

    # Plot ROC curve
    roc_auc = plot_roc_curve(y_true, y_probs)
    print(f"ROC AUC: {roc_auc:.4f}")

    # Save the model
    torch.save(model.state_dict(), 'fibonacci_net.pth')
    print("Model saved to fibonacci_model.pth")

if __name__=="__main__":
    main()






