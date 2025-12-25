# NIR_1st_sem
The code for 2 models pipelines for the first semester NIR
##Task
Investigate if source code is vulnerable or not
## Structure
*src_bilstm* contains source code for BiLSTM test and training pipline:

*code → parser.parse() → AST sequence → Word2Vec training → vocab mapping → integer sequences → embedding matrix → BiLSTM → max pooling → sigmoid → binary classification*

*model_bilstm* contains trained model, embedding matrix and word2vec vocabulary

*src_transformer* contains source code for Transformer test and training pipline:

*model_transformer* contains trained model and it's dependencies

