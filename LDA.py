import re
from nltk.corpus import stopwords
from collections import defaultdict
from gensim import corpora
from gensim import models
import gensim
#from itertools import chain

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # remove all the words that more than 20 characters (like website, some numbers combination, etc)
    # only keep one obj_num for continuously words
    string = re.sub(r"[^A-Za-z_]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r" [\W]+ ", " ", string)
    return string

def prepare_data(filename):
    raw_corpus = [clean_str(line) for line in open(filename)]
    stoplist = stopwords.words('english')
    # first extract and split to sentences
    texts = [[word for word in document.lower().split() if word not in stoplist]
             for document in raw_corpus]
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    # calculate the occurence of each word
    # Only keep words that appear more than once
    processed_corpus = [[token for token in text if frequency[token] > 50 and frequency[token] < 150000] for text in texts]
    # associate each word in the corpus with a unique integer ID
    dictionary = corpora.Dictionary(processed_corpus)
    # save more memory
    class MyCorpus(object):
         def __iter__(self):
             for line in processed_corpus:
                 # assume there's one document per line, tokens separated by whitespace
                 yield dictionary.doc2bow(line)

    bow_corpus = MyCorpus()
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    return dictionary,corpus_tfidf


fileName = "full_data/filter_data1"
dictionary, corpus_tfidf = prepare_data(fileName)
lda = gensim.models.ldamodel.LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=100, update_every=1, chunksize=10000, passes=20,minimum_probability=0)
lda_corpus = lda[corpus_tfidf]

print lda.print_topics(100)
# Find the threshold, let's set the threshold to be 1/#clusters,
# To prove that the threshold is sane, we average the sum of all probabilities:
#scores = list(chain(*[[score for topic_id,score in topic] \
#                      for topic in [doc for doc in lda_corpus]]))
#threshold = sum(scores)/len(scores)

for data in lda_corpus:
    max_value,max_index = 0,0
    for i in range(100):
        if data[i][1] > max_value:
            max_index,max_value = i,data[i][1]
    print max_index

