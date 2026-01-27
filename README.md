# Source Code Vulnerability Detection Models
First semester research on ML-based code vulnerability detection
This repository contains two deep learning pipelines for classifying source code as vulnerable or not: BiLSTM and Transformer

## Task
ML-driven solutions usually perform well in the lab, but show poor performance on real-world data.
My purpose is to identify the most effective approach among existing methods for detecting vulnerable code in open-source projects, achieving the best performance on real-world
data

## Structure
**BiLSTM pipeline**

*src_bilstm* contains source code for BiLSTM test and training pipline:

`input code → parser.parse() → AST sequence → Word2Vec training → vocab mapping → integer sequences → embedding matrix → BiLSTM → max pooling → sigmoid → classification`

*model_bilstm* contains trained model, embedding matrix and word2vec vocabulary

**Transformer  pipeline**

*src_transformer* contains source code for Transformer test and training pipline:
`input code → tokenize → set torch format → class weights → codebert-base → logits → argmax → classification`
*model_transformer* contains trained model and it's dependencies

**Results** 
| Model      | Accuracy on test data | F1     | Recall  | Training time|
|------------|------------------------|--------|---------|---------------|
| Transformer| 0.9916                 | 0.8733 | 0.8268  | 23 h          |
| BiLSTM     | 0.8016                 | 0.3830 | 0.7299  | 40 min        |
