# CNN



# This class is for text classification in CNN
python train.py
data needed to be edited:
# output = "support_engineers.csv"
# vector: glove & word2vec
# train = "dataset1_clean_1"
# vector = "w2v.txt"
# seq_size: how many data should people keep
# seq_size = 350
# number of iteration through the data
# num_of_round = 110
files: obj_file (can be empty), support_engineers.csv, w2v file.

# Evaluation code for cnn:
python eval.py
data needed to be edited:
# obj (contains index.pkl, vocabulary.pkl, engineer.pkl), and classifier model ( meta and model itself)
