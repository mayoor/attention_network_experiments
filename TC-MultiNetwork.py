
# coding: utf-8

# In[2]:


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

from keras.layers import Bidirectional, Input, LSTM, Dense, Activation, Conv1D, Flatten, Embedding, MaxPooling1D, Dropout
#from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers
from gensim.models import Word2Vec
from keras.models import Sequential, Model
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from sklearn.utils import shuffle
import pickle
from sklearn.model_selection import train_test_split
from autocorrect import spell
import spacy
from spacy.gold import GoldParse
nlp = spacy.load('en')
import re
from sklearn.utils import shuffle
import keras


# In[3]:


df = pd.read_csv('train.csv')


# In[4]:


pred_cols = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']


# In[5]:


df['total_classes'] = df['toxic']+df['severe_toxic']+df['obscene']+df['threat']+df['insult']+df['identity_hate']


# In[6]:


df['comment_text'] = df['comment_text'].apply(lambda x : x.replace("'", "").replace('"',''))


# In[7]:


def correct_spelling(text):
    words = text_to_word_sequence(text)
    #print (words)
    words = [spell(w) for w in words]
    return " ".join(words)


# In[8]:


#df['comment_text'] = df['comment_text'].apply(lambda x : correct_spelling(x))


# In[9]:


df['comment_text'] = df['comment_text'].apply(lambda x: re.sub('[0-9]','',x))


# In[10]:


def replace_unknown_words_with_UNK(sentence):
    words = text_to_word_sequence(sentence)
    words = [get_word(w) for w in words]
    return " ".join(words)
    


# In[11]:


def get_word(w):
    if w in tokenizer.word_index:
        return w
    else:
        return "unk"


# In[12]:


def train_tokenizer(texts):
    tokenizer = Tokenizer()
    sent_list = texts
    tokenizer.fit_on_texts(sent_list)
    return tokenizer


# In[13]:


def load_glove_embedding(glove_path):
    word2emb = {}
    with open(glove_path, "rb") as fglove:
        for line in fglove:
            cols = line.strip().split()
            word = cols[0]
            embedding = np.array(cols[1:], dtype="float32")
            word2emb[word] = embedding
    return word2emb


# In[14]:


def generate_word2vec(comments):
    sents = [text_to_word_sequence(s) for s in comments]
    vector = Word2Vec(sents, size=100, iter=50, min_count=1)
    return vector


# In[37]:


comment_list = df['comment_text'].tolist()
glove_file = 'glove.840B.300d.txt'
#glove_file = 'glove.6B.100d.txt'
emb_matrix = load_glove_embedding(glove_file)
#emb_matrix = generate_word2vec(comment_list)
max_len = 300
comment_list.append("unk")
tokenizer = train_tokenizer(comment_list)
n_classes = 1


# ### Replacing all the unknown words with UNK. This will have no impact on training as all the words are known

# In[38]:


df['comment_text'] = df['comment_text'].apply(lambda x : replace_unknown_words_with_UNK(x))


# In[17]:


print ("The vocabulary size is: {0}".format(len(tokenizer.word_index)))
print (tokenizer.texts_to_sequences([replace_unknown_words_with_UNK("DFLSDKJFLS ADFSDF was Infosys CEO")]))


# In[18]:


def clean_up(dfin):
    dfin['comment_text'] = dfin['comment_text'].apply(lambda x : str(x).replace("'", "").replace('"',''))
    dfin['comment_text'] = dfin['comment_text'].apply(lambda x: re.sub('[0-9]','',x))
    #dfin['comment_text'] = dfin['comment_text'].apply(lambda x : replace_unknown_words_with_UNK(x))
    return dfin


# In[19]:


class_count = []
for col in pred_cols:
    class_count.append((col,len(df[df[col]==1])))
print (class_count)


# In[20]:


def get_stratified_train(df, oversample=None):
    df_all_toxic = df[np.logical_and(df['toxic'] ==1 , df['total_classes'] ==1)]
    df_all_severe_toxic = df[np.logical_and(df['severe_toxic'] ==1 , df['total_classes'] <=6)]
    df_all_obscene = df[np.logical_and(df['obscene'] ==1 , df['total_classes'] <=6)]
    df_all_threat = df[np.logical_and(df['threat'] ==1 , df['total_classes'] <=6)]
    df_all_insult = df[np.logical_and(df['insult'] ==1 , df['total_classes'] <=6)]
    df_all_identity_hate = df[np.logical_and(df['identity_hate'] ==1 , df['total_classes'] <=6)]
    df_all_rest =df[df['total_classes'] ==0]
    
    print("Counts:- toxic:{0}, severe_toxic:{1}, obscene:{2}, threat:{3}, insult:{4}, identity_hate:{5}, rest:{6}".format(len(df_all_toxic),len(df_all_severe_toxic),len(df_all_obscene),len(df_all_threat),len(df_all_insult),len(df_all_identity_hate), len(df_all_rest)))
    
    X_train_toxic, X_test_toxic = train_test_split(df_all_toxic, test_size=0.10, random_state=42)
    X_train_severe_toxic, X_test_severe_toxic = train_test_split(df_all_severe_toxic, test_size=0.1, random_state=42)
    X_train_obscene, X_test_obscene = train_test_split(df_all_obscene, test_size=0.05, random_state=42)
    X_train_threat, X_test_threat = train_test_split(df_all_threat, test_size=0.05, random_state=42)
    X_train_insult, X_test_insult = train_test_split(df_all_insult, test_size=0.10, random_state=42)
    X_train_identity_hate, X_test_identity_hate = train_test_split(df_all_identity_hate, test_size=0.1, random_state=42)
    X_train_rest, X_test_rest = train_test_split(df_all_rest, test_size=0.10, random_state=42)
    print("Train Counts:- toxic:{0}, severe_toxic:{1}, obscene:{2}, threat:{3}, insult:{4}, identity_hate:{5}, rest:{6}".format(len(X_train_toxic),len(X_train_severe_toxic),len(X_train_obscene),len(X_train_threat),len(X_train_insult),len(X_train_identity_hate), len(X_train_rest)))
    print("Test Counts:- toxic:{0}, severe_toxic:{1}, obscene:{2}, threat:{3}, insult:{4}, identity_hate:{5}, rest:{6}".format(len(X_test_toxic),len(X_test_severe_toxic),len(X_test_obscene),len(X_test_threat),len(X_test_insult),len(X_test_identity_hate), len(X_test_rest)))
    X_train = pd.concat([X_train_toxic, X_train_severe_toxic, X_train_obscene, X_train_threat, X_train_insult, X_train_identity_hate, X_train_rest])
    X_test = pd.concat([X_test_toxic, X_test_severe_toxic, X_test_obscene, df_all_threat, X_test_insult, X_test_identity_hate, X_test_rest[:500]])
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    print (X_train.count(), X_test.count())
    X_train.head()
    print(X_test.head())
    if oversample:
        X_train_toxic_samp = rand_over_sample(40000, X_train_toxic)
        X_train_severe_toxic_samp = rand_over_sample(40000, X_train_severe_toxic)
        X_train_obscene_samp = rand_over_sample(40000, X_train_obscene)
        X_train_threat_samp = rand_over_sample(40000, X_train_threat)
        X_train_insult_samp = rand_over_sample(40000, X_train_insult)
        X_train_identity_hate_samp = rand_over_sample(40000, X_train_identity_hate)
        X_train = pd.concat([X_train_toxic_samp, X_train_severe_toxic_samp, X_train_obscene_samp, X_train_threat_samp, X_train_insult_samp, X_train_identity_hate_samp, X_train_rest])

    return X_train, X_test


# In[21]:


def rand_over_sample(number_of_records, records):
    sample = records.sample(n=number_of_records, replace=True)
    return sample


# In[42]:


def get_train_split(df):
    train, test = train_test_split(df, test_size=0.10, random_state=42)
    train.head()
    XTrain = tokenizer.texts_to_sequences(train.astype(str)['comment_text'].tolist())
    XVal = tokenizer.texts_to_sequences(test.astype(str)['comment_text'].tolist())
    YTrain = np.array(train[['toxic','severe_toxic','obscene','threat','insult','identity_hate']])
    YVal = np.array(test[['toxic','severe_toxic','obscene','threat','insult','identity_hate']])
    ytemp = train['toxic'].astype(str)+train['severe_toxic'].astype(str)+train['obscene'].astype(str)+train['threat'].astype(str)+train['insult'].astype(str)+train['identity_hate'].astype(str)
    YTrainNum = ytemp.apply(lambda x : int(x,2))
    ytemp = test['toxic'].astype(str)+test['severe_toxic'].astype(str)+test['obscene'].astype(str)+test['threat'].astype(str)+test['insult'].astype(str)+test['identity_hate'].astype(str)
    YValNum = ytemp.apply(lambda x : int(x,2))
    return XTrain, XVal, YTrain, YVal, YTrainNum.values, YValNum.values


# In[26]:


def get_weight_matrix_glove(w2vec, tokenizer, emb_dim=100):
    matrix = np.zeros((len(tokenizer.word_index)+1,emb_dim))
    count = 0
    absent_words = []
    for key in tokenizer.word_index:
        if str.encode(key.replace("'", "").replace('"','')) in w2vec:
            matrix[tokenizer.word_index[key]] = w2vec[str.encode(key.replace("'", "").replace('"',''))]
        else:
            count+=1
            absent_words.append(key)
    print (count)
    #print (absent_words)
    return matrix


# In[27]:


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


# In[28]:


"""
This returns CNN based model. There are 6 output classes, all sharing the parameters of a common network.
"""
def get_model(emb_matrix, learning_rate=0.001):
    input = Input(shape=(maxlen,), dtype='int32')
    embedding = Embedding( input_dim=emb_matrix.shape[0], output_dim=emb_matrix.shape[1], weights=[emb_matrix],input_length=maxlen,trainable=True)

    sequence_input = embedding(input)
    x = Conv1D(64, 3, activation='relu')(sequence_input)
    x = MaxPooling1D(2)(x)
    x = Conv1D(128, 3, activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(256, 3, activation='relu')(x)
    x = MaxPooling1D(2)(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation='relu')(x)

    #toxic	severe_toxic	obscene	threat	insult	identity_hate
    preds_toxic = Dense(n_classes, activation='sigmoid')(x)
    preds_servere_toxic = Dense(n_classes, activation='sigmoid')(x)
    preds_obscene = Dense(n_classes, activation='sigmoid')(x)
    preds_threat = Dense(n_classes, activation='sigmoid')(x)
    preds_insult = Dense(n_classes, activation='sigmoid')(x)
    preds_identity_hate = Dense(n_classes, activation='sigmoid')(x)
    
    model = Model(input,[preds_toxic, preds_servere_toxic, preds_obscene, preds_threat, preds_insult, preds_identity_hate])
    #model.add(Activation('softmax'))
    sgd = optimizers.SGD(lr=learning_rate, clipvalue=0.5)
    model.compile(loss='mse', optimizer=sgd,metrics=['accuracy'])

    print (model.summary())

    return model


# In[29]:


"""
This returns LSTM based model. There are 6 output classes, all soft sharing the parameters of a common network.
"""
def get_model_soft_sharing_lstm(emb_matrix, learning_rate=0.001):
    input = Input(shape=(maxlen,), dtype='int32')
    embedding = Embedding( input_dim=emb_matrix.shape[0], output_dim=emb_matrix.shape[1], weights=[emb_matrix],input_length=maxlen,trainable=True)

    sequence_input = embedding(input)
    x = Bidirectional(LSTM(128,return_sequences=True))(sequence_input)
    x = Bidirectional(LSTM(128,return_sequences=False))(x)

    x = Dense(256, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x1 = Dense(64, activation='relu')(x)
    x2 = Dense(64, activation='relu')(x)
    x3 = Dense(64, activation='relu')(x)
    x4 = Dense(64, activation='relu')(x)
    x5 = Dense(64, activation='relu')(x)
    x6 = Dense(64, activation='relu')(x)

    #toxic	severe_toxic	obscene	threat	insult	identity_hate
    preds_toxic = Dense(n_classes, activation='sigmoid')(x1)
    preds_servere_toxic = Dense(n_classes, activation='sigmoid')(x2)
    preds_obscene = Dense(n_classes, activation='sigmoid')(x3)
    preds_threat = Dense(n_classes, activation='sigmoid')(x4)
    preds_insult = Dense(n_classes, activation='sigmoid')(x5)
    preds_identity_hate = Dense(n_classes, activation='sigmoid')(x6)
    
    model = Model(input,[preds_toxic, preds_servere_toxic, preds_obscene, preds_threat, preds_insult, preds_identity_hate])
    #model.add(Activation('softmax'))
    adam = optimizers.Adam(lr=learning_rate)
    model.compile(loss='mse', optimizer=adam,metrics=['accuracy'])

    print (model.summary())

    return model


# In[35]:


"""
This returns LSTM based model. There are 6 output classes, all soft sharing the parameters of a common network.
"""
def get_model_soft_sharing_lstm_singleoutput(emb_matrix, learning_rate=0.001, n_classes=1, loss='binary_crossentropy'):
    input = Input(shape=(maxlen,), dtype='int32')
    embedding = Embedding( input_dim=emb_matrix.shape[0], output_dim=emb_matrix.shape[1], weights=[emb_matrix],input_length=maxlen,trainable=True)

    sequence_input = embedding(input)
    x = Bidirectional(LSTM(128,return_sequences=True))(sequence_input)
    x = Bidirectional(LSTM(128,return_sequences=False))(x)

    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    preds = Dense(n_classes, activation='sigmoid')(x)

    #toxic	severe_toxic	obscene	threat	insult	identity_hate
   
    
    model = Model(input,preds)
    #model.add(Activation('softmax'))
    adam = optimizers.Adam(lr=learning_rate)
    model.compile(loss=loss, optimizer=adam,metrics=['accuracy'])
    #model.compile(loss='mse', optimizer=adam,metrics=['accuracy'])

    print (model.summary())

    return model


# In[31]:


# Callbacks are passed to the model fit the `callbacks` argument in `fit`,
# which takes a list of callbacks. You can pass any number of callbacks.
callbacks_list = [
    # This callback will interrupt training when we have stopped improving
    keras.callbacks.EarlyStopping(
        # This callback will monitor the validation accuracy of the model
        monitor='val_loss',
        # Training will be interrupted when the accuracy
        # has stopped improving for *more* than 1 epochs (i.e. 2 epochs)
        patience=10,
    ),
    # This callback will save the current weights after every epoch
    keras.callbacks.ModelCheckpoint(
        filepath='/Users/mayoor/dev/kaggle/tc/models/tc.h5',  # Path to the destination model file
        # The two arguments below mean that we will not overwrite the
        # model file unless `val_loss` has improved, which
        # allows us to keep the best model every seen during training.
        monitor='val_loss',
        save_best_only=True,
    ),
    
       keras.callbacks.ReduceLROnPlateau(
           # This callback will monitor the validation loss of the model
           monitor='val_loss',
           # It will divide the learning by 10 when it gets triggered
           factor=0.1,
           # It will get triggered after the validation loss has stopped improving
           # for at least 10 epochs
           patience=3,
) ,

    keras.callbacks.TensorBoard(
        # Log files will be written at this location
        log_dir='/Users/mayoor/dev/kaggle/tc/logs',
        # We will record activation histograms every 1 epoch
        histogram_freq=1,
        # We will record embedding data every 1 epoch
        embeddings_freq=1,
) 

]


# #### Use the X_train_* to create XTrains and YTrains.

# In[32]:


"""
Call this method if you are not directly using the df to split into test and train
"""
def get_xtrain_Ytrain(X_train):
    X_train = shuffle(X_train)
    XTrain = tokenizer.texts_to_sequences(X_train.astype(str)['comment_text'].tolist())
    YTrain_toxic = np.array(X_train['toxic'].tolist())
    YTrain_severe_toxic = np.array(X_train['severe_toxic'].tolist())
    YTrain_obscene = np.array(X_train['obscene'].tolist())
    YTrain_threat = np.array(X_train['threat'].tolist())
    YTrain_insult = np.array(X_train['insult'].tolist())
    YTrain_identity_hate = np.array(X_train['identity_hate'].tolist())
    #YTrain = [YTrain_toxic, YTrain_severe_toxic, YTrain_obscene, YTrain_threat, YTrain_insult, YTrain_identity_hate]
    YTrain = np.array(X_train[['toxic','severe_toxic','obscene','threat','insult','identity_hate']])
    #print(YTrain.shape)
    X_train.head()
    return XTrain, YTrain


# In[33]:


def get_xval_Yval(X_test):
    XVal = tokenizer.texts_to_sequences(X_test.astype(str)['comment_text'].tolist())
    #print(XTrain[0:10],(X_test.astype(str)['comment_text'][0:10]))
    YVal_toxic = np.array(X_test['toxic'].tolist())
    YVal_severe_toxic = np.array(X_test['severe_toxic'].tolist())
    YVal_obscene = np.array(X_test['obscene'].tolist())
    YVal_threat = np.array(X_test['threat'].tolist())
    YVal_insult = np.array(X_test['insult'].tolist())
    YVal_identity_hate = np.array(X_test['identity_hate'].tolist())
    #YVal = [YVal_toxic, YVal_severe_toxic, YVal_obscene, YVal_threat, YVal_insult, YVal_identity_hate]
    YVal = np.array(X_test[['toxic','severe_toxic','obscene','threat','insult','identity_hate']])
    return XVal, YVal


# In[43]:


XTrain, XVal, YTrain, YVal, YTrainNum, YValNum = get_train_split(df)


# In[77]:


maxlen = 300
#final_emb_matrix = get_weight_matrix_local(emb_matrix, tokenizer, 100)
final_emb_matrix = get_weight_matrix_glove(emb_matrix, tokenizer, 300)
#model = get_model(final_emb_matrix, learning_rate=0.001)
model = get_model_soft_sharing_lstm_singleoutput(final_emb_matrix, learning_rate=0.01, n_classes=64, loss='categorical_crossentropy')
#model = get_model_soft_sharing_lstm(final_emb_matrix, learning_rate=0.001)


# In[72]:


ytrainbi = np.zeros((len(YTrainNum),64))
ytestbi = np.zeros((len(YValNum),64))
for i in range(len(YTrainNum)):
    ytrainbi[i][YTrainNum[i]] = 1
for i in range(len(YValNum)):
    ytestbi[i][YValNum[i]] = 1


# In[71]:


print(YTrainNum[9000], YTrain[9000])
ytrainbi[9000]


# In[78]:


model.fit(pad_sequences(XTrain, maxlen),ytrainbi ,batch_size=256, epochs=10, verbose=1, validation_data=(pad_sequences(XVal, maxlen), ytestbi), callbacks=callbacks_list)


# In[ ]:


model.load_weights('/Users/mayoor/dev/kaggle/tc/models/tc.h5')
model.evaluate(pad_sequences(XVal, maxlen), YVal, batch_size=128)
#print("\nTest score: %.3f, accuracy: %.3f" % (v_score, v_acc))


# In[ ]:


test_df = pd.read_csv('test.csv')
test_df = clean_up(test_df)
test_comments = test_df['comment_text'].astype(str).tolist()
XTest = tokenizer.texts_to_sequences(test_comments)
print (test_df.columns)
test_df.head()


# In[ ]:


predictions = model.predict(pad_sequences(XTest, maxlen))


# In[ ]:


predictions.shape


# In[ ]:


predicted_df = pd.DataFrame(columns=['id','toxic','severe_toxic','obscene','threat','insult','identity_hate'])
predicted_df['id'] = test_df['id']
for i, k in enumerate(pred_cols):
    predicted_df[k] = predictions[:,i]
predicted_df.head()


# In[ ]:


predicted_df.to_csv('first_submission.csv',index=False, header=True)


# In[ ]:


print(test_df[test_df['id']==361592343415])
predicted_df[predicted_df['id']==361592343415]


# In[ ]:


print(test_df[test_df['id']==361543686278])
predicted_df[predicted_df['id']==361543686278]


# In[ ]:


pd.options.display.max_colwidth = 600
print(test_df[test_df['id']==361544361532]['comment_text'])
predicted_df[predicted_df['id']==361544361532]

