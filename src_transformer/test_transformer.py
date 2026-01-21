import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = r"/model_transformer"
MAX_LEN = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    use_safetensors=True
).to(device)

model.eval()
print(model.classifier.out_proj.weight.abs().mean())

print(model.config.num_labels)


def predict_vulnerability(code: str, threshold=0.02):
    inputs = tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

    vuln_prob = probs[0, 1].item()
    label = int(vuln_prob > threshold)


    return label, vuln_prob

code_sample = r"""
inline static int is_write_comp_null(gnutls_session_t session)\n{\n if (session->security_parameters.write_compression_algorithm ==\n\tGNUTLS_COMP_NULL)\n\treturn 0;\n\n return 1;\n}
"""

label, prob = predict_vulnerability(code_sample)

print("Vulnerable" if label == 1 else "Non-vulnerable")
print(f"Vulnerability probability: {prob:.4f}")

