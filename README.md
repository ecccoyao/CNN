# CNN



# This class is for text classification in CNN
python train.py
data needed to be edited:
1. output = "support_engineers.csv"
2. vector: glove & word2vec
3. train = "dataset1_clean_1"
4. vector = "w2v.txt"
5. seq_size: how many data should people keep
6. seq_size = 350
7. number of iteration through the data
8. num_of_round = 110
files: obj_file (can be empty), support_engineers.csv, w2v file.

# Evaluation code for cnn:
python eval.py
data needed to be edited:
files: obj (contains index.pkl, vocabulary.pkl, engineer.pkl), and classifier model ( meta and model itself)
