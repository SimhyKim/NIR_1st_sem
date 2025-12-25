import os
import torch
import torch.nn as nn
import numpy as np
import pickle

EMBEDDING_DIM = 100
HIDDEN_DIM = 128
LSTM_LAYERS = 2
DROPOUT = 0.3
MAX_SEQ_LEN = 500

#1. Re-define BiLSTMClassifier
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
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        final_feature_map, _ = torch.max(lstm_out, dim=1)
        return self.fc(final_feature_map)

# 2. Load saved
model_path = "/model_bilstm/bilstm_vulnerability_model.pth"
vocab_path = "/model_bilstm/vocab.pkl"
emb_path = "/model_bilstm/embedding_matrix.npy"

with open(vocab_path, "rb") as f:
    vocab = pickle.load(f)
embedding_matrix = np.load(emb_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BiLSTMClassifier(embedding_matrix, HIDDEN_DIM, LSTM_LAYERS, DROPOUT).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

#3. Set up Tree-sitter parser
os.makedirs("parsers", exist_ok=True)
from tree_sitter_languages import get_language
from tree_sitter import Parser

cpp_lang = get_language('cpp')
parser = Parser()
parser.set_language(cpp_lang)

#4. AST sequence extractor
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

#5. Vectorizer 
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

def vectorize_sequence(seq, vocab, max_len):
    vectorized = [vocab.get(token, vocab[UNK_TOKEN]) for token in seq]
    if len(vectorized) < max_len:
        vectorized += [vocab[PAD_TOKEN]] * (max_len - len(vectorized))
    else:
        vectorized = vectorized[:max_len]
    return vectorized

#6. Input C++ code 
code = """
static int cirrus_bitblt_videotovideo_patterncopy(CirrusVGAState * s)\n{\n return cirrus_bitblt_common_patterncopy(s,\n\t\t\t\t\t s->vram_ptr +\n (s->cirrus_blt_srcaddr & ~7));\n}
"""

#7. Preprocess and predict 
ast_seq = extract_ast_sequence(code)
if not ast_seq:
    raise ValueError("Failed to parse code into AST sequence.")

input_vec = vectorize_sequence(ast_seq, vocab, MAX_SEQ_LEN)
input_tensor = torch.tensor([input_vec], dtype=torch.long).to(device)  # Add batch dim


with torch.no_grad():
    logits = model(input_tensor).squeeze()
    prob = torch.sigmoid(logits).item()
    pred = int(prob > 0.15)


print(f"Code snippet:\n{code}\n")
print(f"Prediction: {'Non-vulnerable' if pred == 1 else 'Vulnerable'}")
print(f"Probability: {prob:.4f}")
