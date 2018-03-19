from keras.layers import Bidirectional, Input, LSTM, Dense, Activation, Conv1D, Flatten, Embedding, MaxPooling1D, Dropout
#from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers
from keras.models import Sequential, Model
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from sklearn.utils import shuffle
import pickle
from sklearn.model_selection import train_test_split
import re
from sklearn.utils import shuffle
import keras
import joblib
import sys
from collections import defaultdict, Counter
from nltk.tokenize import sent_tokenize, word_tokenize
#import spacy

#nlp = spacy.load('en')

def replace_unknown_words_with_UNK(sentence, tokenizer):
    words = word_tokenize(sentence)
    words = [get_word(w, tokenizer) for w in words]
    return " ".join(words)

def get_word(w, tokenizer):
    if w in tokenizer.word_index:
        return w
    else:
        return "unk"

def train_tokenizer(texts):
    tokenizer = Tokenizer(num_words=40000)#Tokenizer(filters='#%&()*+-/<=>?@[\\]^_`{|}~\t\n', num_words=30000)
    sent_list = texts
    #sent_list.append("unk")
    #sent_list = [" ".join([w.text for w in nlp(s)]) for s in sent_list]
    #sent_list = [" ".join(word_tokenize(s)) for s in sent_list]
    tokenizer.fit_on_texts(sent_list)
    return tokenizer


def load_glove_embedding(glove_path):
    word2emb = {}
    with open(glove_path, "rb") as fglove:
        for line in fglove:
            cols = line.strip().split()
            word = cols[0]
            embedding = np.array(cols[1:], dtype="float32")
            word2emb[word] = embedding
    return word2emb


def compute_word_freq(comment_list, tokenizer):
    counter = defaultdict(int)
    for comment in comment_list:
        words = text_to_word_sequence(comment)
        for w in words:
            counter[w]+=1
    return counter

def get_restricted_vocab(counter,max_count):
    restricted_vocab = [counter[k] for k in counter if counter[k]>= max_count]
    return restricted_vocab

def restricted_word_freq_whole_list(comment_list, maxfreq, word_counter, replacement_word):
    refined_list = []
    for comment in comment_list:
        words = text_to_word_sequence(text)
        refined_sent = restricted_word_for_sent(comment, maxfreq, word_counter, replacement_word)
        refined_list.append(refined_sent)
    return refined_list

def restricted_word_for_sent(sent, maxfreq, word_counter, replacement_word=None):
    words = text_to_word_sequence(sent)
    refined_sent = sent
    if words:
        if replacement_word:
            refined_sent = " ".join([w if word_counter[w] >= maxfreq else replacement_word for w in words ])
        else:
            refined_sent = " ".join([w for w in words if word_counter[w] >= maxfreq])
    return refined_sent

def save(embmatrix, tokenizer, token_file='tokenizer', embedding_file='embedding'):
    joblib.dump(embmatrix, embedding_file)
    joblib.dump(tokenizer, token_file)

def clean_up(dfin):
    #dfin['comment_text'] = dfin['comment_text'].apply(lambda x : str(x).replace("'", "").replace('"',''))
    dfin['comment_text'] = dfin['comment_text'].apply(lambda x: re.sub('[0-9]','',x))
    #dfin['comment_text'] = dfin['comment_text'].apply(lambda x : replace_unknown_words_with_UNK(x))
    return dfin

def get_weight_matrix_glove(w2vec, tokenizer, emb_dim=100, vocab_freq_restriction=-1, counter=None):
    restricted_vocab = None
    if counter:
        print('Len of counter {0}'.format(len(counter)))
    if vocab_freq_restriction > 0 and counter:
        restricted_vocab = get_restricted_vocab(counter,vocab_freq_restriction)

    matrix = np.zeros((len(tokenizer.word_index)+1,emb_dim))
    count = 0
    absent_words = []
    for key in tokenizer.word_index:
        if str.encode(key.replace("'", "").replace('"','')) in w2vec :
            if counter and counter[key] < vocab_freq_restriction:
                continue
            matrix[tokenizer.word_index[key]] = w2vec[str.encode(key.replace("'", "").replace('"',''))]
        else:
            count+=1
            absent_words.append(key)
    print (count)
    #print (absent_words)
    return matrix


def get_weight_matrix_local(w2vec, tokenizer, emb_dim=100):
    matrix = np.zeros((len(tokenizer.word_index)+1,emb_dim))
    count = 0
    absent_words = []
    for key in tokenizer.word_index:
        if key.replace("'", "").replace('"','') in w2vec:
            matrix[tokenizer.word_index[key]] = w2vec[key.replace("'", "").replace('"','')]
        else:
            count+=1
            absent_words.append(key)
    print (count)
    #print (absent_words)
    return matrix

def correct_spelling(text):
    words = text_to_word_sequence(text)
    #print (words)
    words = [spell(w) for w in words]
    return " ".join(words)

def define_tokenizer_embedding(train_file='train.csv',glove_file = 'glove.840B.300d.txt', embedding_size=300, token_file='tokenizer', embedding_file='embedding', limit_freq=0  ):
    df = pd.read_csv('train.csv')
    pred_cols = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
    df['total_classes'] = df['toxic']+df['severe_toxic']+df['obscene']+df['threat']+df['insult']+df['identity_hate']
    #df['comment_text'] = df['comment_text'].apply(lambda x : x.replace("'", "").replace('"',''))
    df['comment_text'] = df['comment_text'].apply(lambda x: re.sub('[0-9]','',x))
    emb_matrix = load_glove_embedding(glove_file)
    comment_list = df['comment_text'].tolist()
    tokenizer = train_tokenizer(comment_list)
    counter = None
    if limit_freq > 0:
        print("The limit frequency is {0}".format(limit_freq))
        counter = compute_word_freq(comment_list, tokenizer)
        print("The counter len is: {0}".format(len(counter)))
        df['comment_restricted'] = df['comment_text'].apply(lambda x: restricted_word_for_sent(x, limit_freq, counter, replacement_word=None))
        df.to_csv('train_cleanedup.csv', index=False, header=True)
        tokenizer = train_tokenizer(df['comment_restricted'].tolist())
    print("Tokenizer vocabular after cleanup is {0}".format(len(tokenizer.word_index)))
    final_emb_matrix = get_weight_matrix_glove(emb_matrix, tokenizer, embedding_size, limit_freq, counter)
    save(final_emb_matrix, tokenizer, token_file=token_file, embedding_file=embedding_file)

def get_undersample_training_and_split(data, majority_sample_size, minority_sample_size, test_maj, test_min, comment_col):
    majority =df[df['total_classes'] ==0]
    minority = df[df['toxic'] ==1]
    all_data = majority.sample(n=majority_sample_size, replace=False)
    minority_data = minority.sample(n=minority_sample_size, replace=False)
    training_data = pd.concat([all_data, minority_data])
    test_data = pd.concat([majority.sample(n=test_maj, replace=False),minority.sample(n=test_min, replace=False)])
    return training_data, test_data

def sent_counter(all_docs, maxwords=25):
    counter = defaultdict(int)
    for doc in all_docs:
        sents = sentence_tokenizer(doc, maxwords)
        counter[len(sents)] += 1
    ctc = Counter(counter)
    std = sorted(ctc)
    return std


"""
Splits the document into sentences with max words each. Does not use any NLP techinique to
tokenize sentence. The sentences are fixed width. The spill over goes to the next sentence.
"""
def sentence_tokenizer(doc, max_words):
    #words = word_tokenize(doc)
    words = text_to_word_sequence(doc)
    #words = [w.text for w in nlp(doc)]
    #print(words)
    word_list = []
    for i in range(0,len(words), max_words):
        word_list.append(" ".join(words[i:i+max_words]))
    return word_list

"""
This methods tokenizes the input list of documents into sentence by max words array.
The method does not use an NLP technique.
"""
def pad_sentences(all_docs, max_sent, max_words, tokenizer):
    padded_sentences = []
    #for it in range(max_sent):
    #    padded_sentences.append([])
    for doc in all_docs:
        sentences = sentence_tokenizer(doc, max_words)
        if len(sentences) < max_sent:
            for i in range (max_sent-len(sentences)):
                sentences.append("RRRRRRRRRRRRRRRRRRR")
        if len(sentences) > max_sent:
            sentences = sentences[:max_sent]
        #print (sentences, len(sentences))
        seq_sentences = tokenizer.texts_to_sequences(sentences)
        #print (seq_sentences)
        sent_padded = pad_sequences(seq_sentences,max_words).tolist()
        #for ind, sent in enumerate(sent_padded):
        padded_sentences.append(sent_padded)

    data = np.array(padded_sentences)
    return data#.reshape(data.shape[1],max_sent, max_words)

"""
prepares the document for word level attention
"""
def get_padded_words(sentences, word_len, tokenizer):
    t = tokenizer.texts_to_sequences(sentences)
    return pad_sequences(t, word_len)

"""
This methods tokenizes the input list of documents into sentence by max words array.
The method uses NLP technique to tokenize the sentence.
"""
def pad_sentences_sent(all_docs, max_sent, max_words, tokenizer):
    padded_sentences = []
    #for it in range(max_sent):
    #    padded_sentences.append([])
    for doc in all_docs:
        sentences = sent_tokenize(doc)
        sents = []
        for s in sentences:
            sents += sentence_tokenizer(s, max_words)

        if len(sents) < max_sent:
            for i in range (max_sent-len(sents)):
                sents.append("RRRRRRRRRRRRRRRRRRR")
        if len(sents) > max_sent:
            sents = sents[:max_sent]
        #print (sents, len(sentences))

        seq_sentences = tokenizer.texts_to_sequences(sents)
        #print (seq_sentences)
        sent_padded = pad_sequences(seq_sentences,max_words).tolist()
        #for ind, sent in enumerate(sent_padded):
        padded_sentences.append(sent_padded)

    data = np.array(padded_sentences)
    return data


def get_train_split(df):
    train, test = train_test_split(df, test_size=0.10, random_state=42)
    train.head()
    XTrain = tokenizer.texts_to_sequences(train.astype(str)[comment_col].tolist())
    XVal = tokenizer.texts_to_sequences(test.astype(str)[comment_col].tolist())
    YTrain = np.array(train[pred_cols])
    YVal = np.array(test[pred_cols])
    #ytemp = train['toxic'].astype(str)+train['severe_toxic'].astype(str)+train['obscene'].astype(str)+train['threat'].astype(str)+train['insult'].astype(str)+train['identity_hate'].astype(str)
    #YTrainNum = ytemp.apply(lambda x : int(x,2))
    #ytemp = test['toxic'].astype(str)+test['severe_toxic'].astype(str)+test['obscene'].astype(str)+test['threat'].astype(str)+test['insult'].astype(str)+test['identity_hate'].astype(str)
    #YValNum = ytemp.apply(lambda x : int(x,2))
    return XTrain, XVal, YTrain.reshape(len(XTrain),), YVal.reshape(len(XVal),)


def get_word_reverse_lookup(tokenizer):
   lookup_words = {tokenizer.word_index[k]:k for k in tokenizer.word_index}
   return lookup_words

"""
This method takes document and reverse lookup dictionary and generates a sentence by word matrix equivalent to
the input matrix, but instead of word index number, the matrix contains the actual words
"""
def readable_pad_sent(doc, max_sent, max_words, lookup_words):
    padded_sentences = []
    #for it in range(max_sent):
    #    padded_sentences.append([])
    sentences = sentence_tokenizer(doc, max_words)#sent_tokenize(doc)
    if len(sentences) < max_sent:
        for i in range (max_sent-len(sentences)):
            sentences.append("RRRRRRRRRRRRRRRRRRR")
    if len(sentences) > max_sent:
        sentences = sentences[:max_sent]
    #print (sentences, len(sentences))
    seq_sentences = tokenizer.texts_to_sequences(sentences)
    print (seq_sentences)
    sent_padded = pad_sequences(seq_sentences,max_words).tolist()
    #for ind, sent in enumerate(sent_padded):
    for item in sent_padded:
        s = [lookup_words[i] if i !=0 else " " for i in item ]
        padded_sentences.append(s)

    return padded_sentences

if __name__ == "__main__":
    limit_freq = 0
    embedding_size = 300
    if len(sys.argv) >= 2:
        embedding_size = int(sys.argv[1])
        if len(sys.argv) == 3:
            limit_freq =  int(sys.argv[2])
        print('Starting processing with configuration - Embedding_size:{0} and Limit_Freq:{1}'.format(embedding_size, limit_freq))
        if  int(sys.argv[1])==300:
            define_tokenizer_embedding(embedding_size=embedding_size, limit_freq=limit_freq)
        else:
            define_tokenizer_embedding(train_file='train.csv',glove_file = 'glove.6B.100d.txt', embedding_size=embedding_size, token_file='tokenizer_100', embedding_file='embedding_100', limit_freq=limit_freq )
