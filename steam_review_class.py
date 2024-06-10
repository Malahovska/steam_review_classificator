from google.colab import files
import sys
import importlib
import os
import pandas as pd
import requests
import matplotlib.pyplot as plt
import zipfile
import seaborn as sns
import torch
import torch.nn.functional as nn
import torchtext
import transformers
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification, AdamW
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

"""# Homework Environment Setup"""

# Check if CUDA is available
if(torch.cuda.is_available()):
  DEVICE = torch.device('cuda:0')
else:
  DEVICE = torch.device('cpu')

"""# Data Upload and Extraction"""

uploaded = files.upload()  # Upload the train file
with zipfile.ZipFile('train.zip', 'r') as zip_file:  # Extract the train.csv file
    zip_file.extract('train.csv')

"""# Data Preparation"""

# Read the dataset and add labels
df = pd.read_csv('train.csv', header=None, names=['app_id', 'app_name', 'review_text', 'review_score', 'review_votes'])

# Filter out unwanted rows and adjust the data
df = df[df['app_name'] != "The Long Dark"]
df['review_score'] = df['review_score'].replace(-1, 0)
df = df.dropna()

# Define dataset sizes, no more than 45000 due to resource limits
train_size = int(45000 * 0.7)
valid_size = int(45000 * 0.1)
test_size = int(45000 * 0.2)

print("Train size:", train_size, "\nValidation size:", valid_size, "\nTest size:", test_size)

# Split the data for training, validation, and testing
divide = df.sample(n=45000, random_state=42) # More than this will exceed memory limits

train_texts = divide.iloc[:train_size]['review_text'].values
train_score = divide.iloc[:train_size]['review_score'].values
valid_texts = divide.iloc[train_size:train_size+valid_size]['review_text'].values
valid_score = divide.iloc[train_size:train_size+valid_size]['review_score'].values
test_texts = divide.iloc[train_size + valid_size:train_size + valid_size + test_size]['review_text'].values
test_score = divide.iloc[train_size + valid_size:train_size + valid_size + test_size]['review_score'].values

# Convert arrays to lists
train_texts_list = train_texts.tolist()
valid_texts_list = valid_texts.tolist()
test_texts_list = test_texts.tolist()

"""# Initialize tokenizer for input encodings"""

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english') # Get the tokenizer object using pre-trained distilBERT model

# Create input encodings for training, validation, and testing
train_encodings = tokenizer(train_texts_list, truncation=True, padding=True)
valid_encodings = tokenizer(valid_texts_list, truncation=True, padding=True)
test_encodings = tokenizer(test_texts_list, truncation=True, padding=True)

"""# Dataset and Model Class Creation"""

# Create the SteamDataset class
class SteamDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create the training, validation, and test datasets
train_dataset = SteamDataset(train_encodings, train_score)
valid_dataset = SteamDataset(valid_encodings, valid_score)
test_dataset = SteamDataset(test_encodings, test_score)

# Create DataLoader objects to iterate through the datasets in batches
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

"""# Model Loading"""

# Load model weights from Google Drive
# Can be used without, but it doesn't always work
file_id = 'path to your weights'
destination = 'model_weights.pth'
!gdown --id $file_id --output $destination

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english') # Load the distilBERT model
model.load_state_dict(torch.load('model_weights.pth', map_location=DEVICE))
model.to(DEVICE) # Move the model to the device
model.train() # Set the model to training mode
optim = AdamW(model.parameters(), lr=5e-5) # Create the optimizer

"""# Training"""

TRAIN = False  # Determine whether to train, for the first time set to true

if TRAIN:
    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch in train_loader:

            # Get input data
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            # Clear previous iteration gradients
            optim.zero_grad()

            # Perform model forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            # Extract loss from output
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()

            # Perform optimizer step
            optim.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{3}, Average Loss: {avg_loss}")

    # Save the weights
    torch.save(model.state_dict(), 'model_weights.pth')
else:
    # If TRAIN is False, load weights from Google Drive
    if not os.path.exists('model_weights.pth'):
        !wget --output-document #your path to weights
    model.load_state_dict(torch.load('model_weights.pth'))

"""# Accuracy Evaluation"""

# Define a function to plot the confusion matrix
def plot_confusion_matrix(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    model.to(device)

    # Create empty lists to store model predictions and true labels
    all_preds = []
    all_labels = []

    # Execute code without backpropagation
    with torch.no_grad():
        for batch in data_loader:

            # Get input data from batch and move them to the specified device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Perform model prediction using input data
            outputs = model(input_ids, attention_mask=attention_mask)

            # Choose the final output label using argmax function
            _, predicted_labels = torch.max(outputs.logits, 1)

            # Update lists with predicted labels and true labels
            all_preds.extend(predicted_labels.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds) # Compute confusion matrix

    # Create a visual representation of the confusion matrix using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

plot_confusion_matrix(model, test_loader, DEVICE)

# Define a function to compute F1 score, similar to the previous function for the confusion matrix
def compute_f1_score(model, data_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted_labels = torch.max(outputs.logits, dim=1)

            all_predictions.extend(predicted_labels.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_predictions)
    return f1
print("F1 score:", compute_f1_score(model, test_loader, DEVICE))
```