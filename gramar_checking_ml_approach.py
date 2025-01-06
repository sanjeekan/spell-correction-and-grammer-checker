import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
import pandas as pd

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Tamil grammar correction dataset
file_path = "/content/drive/MyDrive/Tamil_Grammar_Correction_Dataset.xlsx"
df = pd.read_excel(file_path)

# Preprocess: Label sentences (1: Correct, 0: Incorrect)
df["label"] = df["Type of Error"].apply(lambda x: 1 if x == "Correct" else 0)

# Split dataset into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["Sentence"], df["label"], test_size=0.2, random_state=42
)

# Load mBERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# Tokenize sentences
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)

# Create a custom dataset
class TamilDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

# Create datasets
train_dataset = TamilDataset(train_encodings, train_labels.tolist())
val_dataset = TamilDataset(val_encodings, val_labels.tolist())

# Load mBERT model
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
model.to(device)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

# Evaluation function
def evaluate(sentence):
    model.eval()
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probs, dim=1).item()
    if prediction == 1:
        return f"'{sentence}' is grammatically correct."
    else:
        return f"'{sentence}' is incorrect. Please review grammar."

# Test the model
test_sentence_1 = "அவர் செல்கிறார் நான் வெற்றி பெற்றேன்."
print(evaluate(test_sentence_1))
