# FEGAT-Cross-Language-Code-Clone-Detection
The source code and models for the paper Cross-Language Code Clone Detection via Flow-Enhanced Graph Attention Network

## Environment
Python 3.8 and TensorFlow 1.15.5

## Model
The model is implemented in `GATSiamese.py`, while others are implementations of baseline models for comparison.
To train the model, run the following command:
```bash
python train.py
To evaluate the model, run the following command:
```bash
python total_test.py
We provide two directories, `models_AtCode` and `models_CodeChef`, which contain the models trained on the AtCoder and CodeChef datasets.
