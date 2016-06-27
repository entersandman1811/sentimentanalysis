"""
This script is what created the dataset pickled.

1) You need to download this file and put it in the same directory as this file.
https://github.com/moses-smt/mosesdecoder/raw/master/scripts/tokenizer/tokenizer.perl . Give it execution permission.

2) Get the dataset from http://ai.stanford.edu/~amaas/data/sentiment/ and extract it in the current directory.

3) Then run this script.
"""

dataset_path='/home/souradeep/Downloads/aclImdb/'
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import numpy
import cPickle as pkl

from collections import OrderedDict

import glob
import os
import operator

from subprocess import Popen, PIPE

tokenizer_cmd = ['./tokenizer.perl', '-l', 'en', '-q', '-']


def tokenize(sentences):

    print 'Tokenizing..',
    text = "\n".join(sentences)
    tokenizer = Popen(tokenizer_cmd, stdin=PIPE, stdout=PIPE)
    tok_text, _ = tokenizer.communicate(text)
    toks = tok_text.split('\n')[:-1]
    print 'Done'

    return toks


def build_dict(path):
    sentences = []
    currdir = os.getcwd()
    os.chdir('%s/pos/' % path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip())
    os.chdir('%s/neg/' % path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip())
    os.chdir(currdir)

    sentences = tokenize(sentences)

    print 'Building dictionary..',



    wordcount = dict()
    for ss in sentences:
        words = ss.strip().lower().split()
        for w in words:
            if w not in wordcount:
                wordcount[w] = 1
            else:
                wordcount[w] += 1


    bow = sorted(wordcount, key=wordcount.get, reverse= True)

    print "The number of unique words in the training set is: " , len(bow)

    return bow[:10000]

def grab_data(path, bow):
    sentences = []
    currdir = os.getcwd()
    os.chdir(path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip())
    os.chdir(currdir)
    sentences = tokenize(sentences)

    seqs = [None] * len(sentences)
    print bow
    for idx, ss in enumerate(sentences):
        words = ss.dvx().lower().split()


        seqs[idx] = [0 if w not in  bow else words.count('w') for w in bow]

    print seqs[0]
    return seqs


def main():
    path = dataset_path
    bow = build_dict(os.path.join(path, 'train'))

    grab_data(path+'train/pos',bow)






    # train_x_pos = grab_data(path+'train/pos', dictionary)
    # train_x_neg = grab_data(path+'train/neg', dictionary)
    # train_x = train_x_pos + train_x_neg
    # train_y = [1] * len(train_x_pos) + [0] * len(train_x_neg)
    #
    # test_x_pos = grab_data(path+'test/pos', dictionary)
    # test_x_neg = grab_data(path+'test/neg', dictionary)
    # test_x = test_x_pos + test_x_neg
    # test_y = [1] * len(test_x_pos) + [0] * len(test_x_neg)
    #
    # f = open('imdb.pkl', 'wb')
    # pkl.dump((train_x, train_y), f, -1)
    # pkl.dump((test_x, test_y), f, -1)
    # f.close()
    #
    # f = open('imdb.dict.pkl', 'wb')
    # pkl.dump(dictionary, f, -1)
    # f.close()

if __name__ == '__main__':
    main()