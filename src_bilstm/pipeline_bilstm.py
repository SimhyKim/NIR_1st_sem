import os

os.makedirs("parsers", exist_ok=True)
assert os.path.exists("parsers/tree-sitter-cpp/src/parser.c"), "parser.c missing!"
from tree_sitter import Language
Language.build_library(
    'parsers/my-languages.so',
    ['parsers/tree-sitter-cpp']
)

from tree_sitter_languages import get_language
from tree_sitter import Parser
import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

cpp_lang = get_language('cpp')  
parser = Parser()
parser.set_language(cpp_lang)

tree = parser.parse(b"int main() { return 0; }")
print(tree.root_node.sexp())

# 1. Configuration & Data Loading
# Configuration based on typical Sequential Model settings in literature
EMBEDDING_DIM = 100    # Standard Word2Vec dimension 
HIDDEN_DIM = 128       # Hidden dimension for LSTM
LSTM_LAYERS = 2        # Number of stacked LSTM layers
DROPOUT = 0.3          
BATCH_SIZE = 64
EPOCHS = 10.           # Experimets shown that >10 epocs drops accuracy
MAX_SEQ_LEN = 500      # Truncate sequences for efficiency



def load_reveal_data(non_vuln_path, vuln_path):
    """
    Loads ReVeal dataset from two JSON files:
    - non-vulnerables.json (label = 0)
    - vulnerables.json     (label = 1)
    """

    def read_json(path, label):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = []
        for item in data:
            code = (
                item.get("code")
                or item.get("func")
                or item.get("function")
            )

            if code is None:
                continue

            records.append({
                "code": code,
                "label": label
            })

        return records

    non_vuln = read_json(non_vuln_path, label=0)
    vuln = read_json(vuln_path, label=1)

    df = pd.DataFrame(non_vuln + vuln)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"Loaded ReVeal dataset: {len(df)} samples")
    print(df["label"].value_counts())

    return df

df = load_reveal_data(
    "/Users/zubarevich.k/Downloads/non-vulnerables.json",
    "/Users/zubarevich.k/Downloads/vulnerables.json"
)

# 2.AST Sequence

def extract_ast_sequence(code: str):
    try:
        tree = parser.parse(bytes(code, "utf8"))
        root = tree.root_node
    except Exception as e:
        print("Parse error:", e)
        return []

    seq = []
    stack = [root]
    while stack:
        node = stack.pop()
        seq.append(node.type)
        stack.extend(reversed(node.children))
    return seq


print("Extracting AST sequences...")
df['ast_seq'] = df['code'].apply(extract_ast_sequence)
df = df[df["ast_seq"].apply(len) > 0].reset_index(drop=True)


X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df['ast_seq'], df['label'], test_size=0.2, random_state=42
)

# 3. Word2Vec

print("Training Word2Vec embedding...")
w2v_model = Word2Vec(
    sentences=X_train_raw,
    vector_size=EMBEDDING_DIM,
    window=5,
    min_count=2,  
    workers=4,
    sg=1          
)




# Create a vocabulary dictionary
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
vocab = {word: i for i, word in enumerate(w2v_model.wv.key_to_index.keys())}
vocab[PAD_TOKEN] = len(vocab)
vocab[UNK_TOKEN] = len(vocab)

# Create embedding matrix
embedding_matrix = np.zeros((len(vocab), EMBEDDING_DIM))
for word in w2v_model.wv.key_to_index:
    embedding_matrix[vocab[word]] = w2v_model.wv[word]
embedding_matrix[vocab[UNK_TOKEN]] = np.random.normal(scale=0.6, size=(EMBEDDING_DIM,))

# Convert df sequences to integer sequences
def vectorize_sequence(seq, vocab, max_len):
    vectorized = [vocab.get(token, vocab[UNK_TOKEN]) for token in seq]
    if len(vectorized) < max_len:
        vectorized += [vocab[PAD_TOKEN]] * (max_len - len(vectorized))
    else:
        vectorized = vectorized[:max_len]
    return vectorized

X_train_vec = [vectorize_sequence(seq, vocab, MAX_SEQ_LEN) for seq in X_train_raw]
X_test_vec = [vectorize_sequence(seq, vocab, MAX_SEQ_LEN) for seq in X_test_raw]


# 4. BiLSTM


class VulnerabilityDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels.values, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

train_dataset = VulnerabilityDataset(X_train_vec, y_train)
test_dataset = VulnerabilityDataset(X_test_vec, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
class BiLSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, lstm_layers, dropout):
        super().__init__()
        num_embeddings, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=False
        )
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=lstm_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim*2, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        final_feature_map, _ = torch.max(lstm_out, dim=1)  
        out = self.fc(final_feature_map)
        return out 


# Initialization of model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BiLSTMClassifier(embedding_matrix, HIDDEN_DIM, LSTM_LAYERS, DROPOUT).to(device)

# 5. Training loop


optimizer = optim.Adam(model.parameters(), lr=0.001)
num_pos = (y_train == 1).sum()
num_neg = (y_train == 0).sum()
pos_weight = num_neg / num_pos  
print("pos_weight =", pos_weight)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))


print(f"Starting training on {device}...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()


        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

# 6. Evaluation

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs).squeeze()
        predicted = (outputs > 0.5).float()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

accuracy = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')

print("\nEvaluation Results on ReVeal:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

import pickle

torch.save(model.state_dict(), "/Users/zubarevich.k/Downloads/model/bilstm_vulnerability_model.pth")

# 2. Save vocab and embedding_matrix for fiuture
with open("/Users/zubarevich.k/Downloads/model/vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

np.save("/Users/zubarevich.k/Downloads/model/embedding_matrix.npy", embedding_matrix)

print("Model, vocab, and embedding matrix saved successfully.")