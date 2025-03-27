import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, lstm_hidden=128, lstm_layers=1, dropout_rate=0.5):
        super(FeedForwardNN, self).__init__()

        # 1D Convolution Layer
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn_conv = nn.BatchNorm1d(16)

        # BiLSTM Layer (input_size = Conv1d out_channels)
        self.lstm = nn.LSTM(input_size=16, hidden_size=lstm_hidden, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True)
        
        # Attention Layer
        self.attention = nn.Linear(lstm_hidden * 2, 1)  # Output size 1 for attention score

        # Fully Connected Layers (Adjust input size based on BiLSTM output)
        self.fc1 = nn.Linear(lstm_hidden * 2, hidden_dim)  # *2 for bidirectional LSTM
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.xavier_uniform_(self.fc4.weight)

    def attention_layer(self, lstm_output):
        """
        Applies attention mechanism on the LSTM output.
        :param lstm_output: Tensor of shape (batch_size, sequence_length, lstm_hidden * 2)
        :return: Context vector after applying attention
        """
        attn_weights = F.softmax(self.attention(lstm_output), dim=1)  # (batch, seq_len, 1)
        context_vector = torch.sum(attn_weights * lstm_output, dim=1)  # Weighted sum: (batch, lstm_hidden * 2)
        return context_vector

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for Conv1d (batch, 1, features)
        x = self.relu(self.bn_conv(self.conv1d(x)))

        # Reshape for LSTM (batch, sequence_length, feature_dim)
        x = x.permute(0, 2, 1)  # Change shape from (batch, channels, features) → (batch, features, channels)

        lstm_output, _ = self.lstm(x)  # BiLSTM output: (batch, seq_len, lstm_hidden * 2)

        # Apply Attention
        x = self.attention_layer(lstm_output)

        # Fully connected layers
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)

        return x



class GenderDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def main():
    # Update paths for VoxCeleb1 data
    train_features_path = r".\preprocess_train.npy"
    test_features_path = r".\preprocess_test.npy"
    train_labels_path = r".\labels_train.npy"
    test_labels_path = r".\labels_test.npy"
    
    # Load data
    print("Loading features and labels...")
    X_train = np.load(train_features_path)
    X_test = np.load(test_features_path)
    y_train = np.load(train_labels_path)
    y_test = np.load(test_labels_path)
    
    print(f"Train features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    print(f"Train labels shape: {y_train.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    # Reshape features
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    # Apply StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)  # Use same scaling parameters
    
    # Model parameters
    input_dim = X_train.shape[1]
    hidden_dim = 512
    output_dim = 1
    
    # Initialize model
    model = FeedForwardNN(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    # Create datasets
    train_dataset = GenderDataset(X_train, y_train)
    test_dataset = GenderDataset(X_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Training and evaluation
    train_accuracies, test_accuracies, train_losses, test_losses = [], [], [], []
    best_test_acc = 0.0  # Initialize the best test accuracy

    for epoch in range(35):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = (outputs > 0).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
        train_acc = correct / total
        train_accuracies.append(train_acc)
        train_losses.append(running_loss / len(train_loader))
        
        # Evaluation
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.float().to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                predicted = (outputs > 0).float()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(targets.cpu().numpy())
                
        test_acc = correct / total
        test_accuracies.append(test_acc)
        test_losses.append(test_loss / len(test_loader))
        
        # Save the model if the test accuracy is the best so far
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')  # Save the model's state dictionary
            print(f"Model saved with Test Accuracy: {best_test_acc:.4f}")
        
        print(f'Epoch [{epoch+1}/35]')
        print(f'Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_acc:.4f}')
        
        scheduler.step()
    
    # Plot results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('gender_classification_results.png')
    
    # Print final classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=['Female', 'Male']))
    
    # Plot confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Female', 'Male'],
                yticklabels=['Female', 'Male'])
    plt.title('VoxCeleb1 Gender Classification Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('voxceleb1_confusion_matrix.png')
    plt.close()
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('voxceleb1_training_history.png')
    plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    main()