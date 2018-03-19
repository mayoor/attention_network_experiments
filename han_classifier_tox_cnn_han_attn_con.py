from keras.layers import Bidirectional, Input, LSTM, Dense, Activation, Conv1D, Flatten, Embedding, GlobalMaxPooling1D, Dropout
from keras.layers import Add, Concatenate, Lambda, Reshape, Permute, Average, Layer, TimeDistributed, Multiply, GRU, BatchNormalization, CuDNNGRU, SpatialDropout1D, GlobalAveragePooling1D, Maximum
#from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers
from keras.models import Sequential, Model
import pandas as pd
import numpy as np
from keras.callbacks import Callback
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from sklearn.utils import shuffle
import pickle
from sklearn.model_selection import train_test_split
import re
from sklearn.utils import shuffle
import keras
import joblib
from keras.utils.vis_utils import plot_model
import keras.backend as K
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
import tokenizer_util as tu
import os
from sklearn.metrics import roc_auc_score
from keras import regularizers
os.environ["CUDA_VISIBLE_DEVICES"]="1"

#nohup python -u han_classifier_tox_cnn_han_attn_con.py > class_output.log &

TRAIN_FILE_PATH = 'train.csv'#'/data/train.csv'
TEST_FILE = 'test.csv'#'/data/test.csv'

TIME_STEPS = 300
BATCH_SIZE = 256
LEARNING_RATE = 0.001
DECAY = 0.001
EPOCH_SIZE = 100

TOKENIZER_FILE = 'tokenizer'
EMBEDDING_FILE = 'embedding'

TENSORFLOW_LOGDIR = 'logs'#'/output/tensorboard_logs'
MODEL_SAVE_PATH = 'models/best_model_attn_cnn_con.h5' #'/output/best_model.h5'
OUTPUT_FILENAME = 'sub_h_n_consolidated_filtered.csv'
SENTENCE = 40
WORDS = 7


df = pd.read_csv(TRAIN_FILE_PATH)

pred_cols = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

df['total_classes'] = df['toxic']+df['severe_toxic']+df['obscene']+df['threat']+df['insult']+df['identity_hate']

comment_col = 'comment_text'

#df[comment_col] = df[comment_col].astype(str).apply(lambda x : x.replace("'", "").replace('"',''))

df[comment_col] = df[comment_col].apply(lambda x: re.sub('[0-9]','',x))

comment_list = df[comment_col].tolist()
n_classes = 1


tokenizer = joblib.load(TOKENIZER_FILE)
final_emb_matrix = joblib.load(EMBEDDING_FILE)
print('Total vocabulary is {0}'.format(final_emb_matrix.shape[0]))


train, test = train_test_split(df, test_size=0.10, random_state=42)

XVal = tokenizer.texts_to_sequences(test.astype(str)[comment_col].tolist())


def ys(dftox, predcols):
    ys = []
    for col in predcols:
        ys.append(np.array(dftox[col].tolist()))
    return ys


def ys_unified(dftox, predcols):
    ys = dftox[predcols].values
    return ys
YTrain = ys_unified(train, pred_cols)
YVal = ys_unified(test, pred_cols)


"""
Attention Layer with works follows the math from https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf
This layer only computes the weights, does not multiply the RNN output with the weights. This layer has to be
followed by a Multiply layer, followed by Reshape, followed by a Lambda for summing.
"""
class ATTNWORD(Layer):
    def __init__(self,output_dim, **kwargs):
        self.output_dim = output_dim
        #self.supports_masking = True
        super(ATTNWORD, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        print('The input shape is: {}'.format(input_shape))
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        self.input_shape_bk = input_shape
        super(ATTNWORD, self).build(input_shape)

    def call(self, x,mask=None):
        print ('kernel shape', self.kernel.shape)
        print ('Input shape', x.shape)
        product = K.dot(x, self.kernel)
        product = K.reshape(product, (-1, self.output_dim, self.input_shape_bk[1]))

        x_norm  = K.softmax(product)
        print ('Norm shape', x_norm.shape)
        x_norm = K.reshape(x_norm, (-1, self.input_shape_bk[1],self.output_dim))

        return x_norm

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def get_config(self):
        return super(ATTNWORD, self).get_config()

"""
A attenion layer, built on the basis of https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf.
Takes care of all the atention compute, Takes array of input - Bidirectional RNN output and the TanH layer output.
Usage ATTNWORD_COMPLETE(1)([tanh_output, rnn_output])
"""
class ATTNWORD_COMPLETE(Layer):
    def __init__(self,output_dim, **kwargs):
        self.output_dim = output_dim
        #self.supports_masking = True
        super(ATTNWORD, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        print('The input shape is: {}'.format(input_shape))
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[0][-1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        self.input_shape_bk = input_shape
        super(ATTNWORD, self).build(input_shape)

    def call(self, x,mask=None):
        print ('kernel shape', self.kernel.shape)
        print ('Input shape', x[0].shape)
        product = K.dot(x[0], self.kernel)
        product = K.reshape(product, (-1, self.output_dim, self.input_shape_bk[0][1]))
        x_norm  = K.softmax(product)
        print ('Norm shape', x_norm.shape)
        x_norm = K.reshape(x_norm, (-1, self.input_shape_bk[0][1],self.output_dim))
        print ('reshaped Norm shape: {0} and hit shape is {1}'.format( x_norm.shape, x[1].shape))
        attn_final = x[1]*x_norm
        print ('Attn final shape', attn_final.shape)
        attn_final = K.reshape(attn_final, (-1, self.input_shape_bk[1][-1], self.input_shape_bk[0][1]))

        attn_final = K.sum(attn_final, axis=2)
        print ('Attn final shape sum', attn_final.shape)
        return attn_final

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[1][-1])

    def get_config(self):
        return super(ATTNWORD_COMPLETE, self).get_config()


"""
This method creates a model with an input of word length, followed by embedding layer and finally GRU,
with output dim as passed in the argument.
"""
def get_word_attention(emb_matrix, word_length, optimizer, nclasses, gru_output_dim=128):
    input = Input(shape=(word_length, ), dtype='int32')
    embedding = Embedding( input_dim=emb_matrix.shape[0], output_dim=emb_matrix.shape[1], weights=[emb_matrix],input_length=word_length,trainable=True, mask_zero=False)
    sequence_input = embedding(input)
    print('embedding: ',sequence_input.shape)
    sequence_input = SpatialDropout1D(0.4)(sequence_input)
    x = Bidirectional(CuDNNGRU(gru_output_dim,return_sequences=True))(sequence_input)
    print('Shape after BD LSTM',x.shape)
    model = Model(input, x)
    return model


"""
This method applies attention only at the word level. The last layer is a sigmoid layer with output of 1.
The output is going to be an array, the number of output is determined by n_classes.
Here the labels are assumed to be independent of each other and probability for each label is independently calculated
using dedicated Attention layer for each.
"""
def attention_words_only(emb_matrix, word_length, n_classes, trainable=True):
    nclasses = n_classes
    preds = []
    attentions_pred = []
    input = Input(shape=(word_length, ), dtype='int32')
    embedding = Embedding( input_dim=emb_matrix.shape[0], output_dim=emb_matrix.shape[1], weights=[emb_matrix],input_length=word_length,trainable=True)
    sequence_input = embedding(input)
    print('embedding: ',sequence_input.shape)
    x = Bidirectional(GRU(50,return_sequences=True))(sequence_input)
    word_vectors = TimeDistributed(Dense(100, activation='tanh'))(x) #TanH layer as required by the paper, is external to the Attn layer.
    print('Shape after word vector',word_vectors.shape)
    h_it = x
    print('Shape after reshape word vector',h_it.shape)

    attn_final_word = [ATTNWORD_COMPLETE(1)([word_vectors, h_it]) for i in range(nclasses)]
    print('ATTN Shape', attn_final_word[0].shape)

    for i in range(nclasses):
        p = Dense(1, activation='sigmoid')(attn_final_word[i])
        preds.append(p)
    model = Model(input, preds)

    return model


def get_sentence_attention(word_model , word_length, sent_length, n_classes):
    #x = Permute((2,1))(si_vects)
    nclasses = 1
    input = Input(shape=(sent_length, word_length ), dtype='int32')
    print(' input to sentence attn network',word_model)
    preds = []
    attentions_pred = []
    #print(output.summary())
    si_vects = TimeDistributed(word_model)(input)
    print('Shape after si_vects', si_vects.shape)
    #u_it = TimeDistributed(TimeDistributed(BatchNormalization()))(si_vects)
    u_it = TimeDistributed(TimeDistributed(Dense(256, activation='tanh')))(si_vects)
    print('Shape after word vector',u_it.shape)
    #u_it = TimeDistributed(TimeDistributed(BatchNormalization()))(u_it)

    #h_it = TimeDistributed(Reshape((100,word_length)))(si_vects)
    #print('Shape after reshape word vector',h_it.shape)

    attn_final_word = [TimeDistributed(ATTNWORD(1))(u_it) for i in range(nclasses)]
    #a_it = Reshape(( word_length, 1))(a_it)
    #h_it = Reshape((word_length, 512))(h_it)
    print('ATTN Shape', attn_final_word[0].shape)
    attn_final_word = [Multiply()([si_vects, attn_final_word[i]]) for i in range(nclasses)]#Multiply()([h_it,a_it])
    print('Multi word Shape', attn_final_word[0].shape)
    attn_final_word = [Reshape((sent_length, 256,word_length))(attn_final_word[i]) for i in range(nclasses)]
    print ('Shape of the att1 is {}'.format(attn_final_word[0].shape))
    attn_final_word = [Lambda(lambda x: K.sum(x, axis=3))(attn_final_word[i]) for i in range(nclasses)]
    output_list = []
    for i in range(nclasses):
        print ('Shape of the lambda word is {}'.format(attn_final_word[i].shape))
        ind_t = 0
        attn_sents_for_all_classes = []
        #attn_final_word[i] = SpatialDropout1D(0.2)(attn_final_word[i])
        x = Bidirectional(CuDNNGRU(128,return_sequences=True))(attn_final_word[i])
        x = SpatialDropout1D(0.2)(x)
        x = BatchNormalization()(x)
        print ("Shape of X-X is {}".format(x.shape))
        u_it = TimeDistributed(Dense(256, activation='tanh'))(x)
        print('Shape after word vector',u_it.shape)
        #h_it = Reshape((100,sent_length))(x)
        attn_final_sent = ATTNWORD(1)(u_it)
        print ('Shape of the sent att is {}'.format(attn_final_sent.shape))
        #attentions_pred.append(attn_final)
        attn_final_sent = Multiply()([x, attn_final_sent])
        print ('Shape of the multi sent att is {}'.format(attn_final_sent.shape))
        attn_final_sent = Reshape((256,sent_length))(attn_final_sent)
        attn_final_sent = Lambda(lambda x: K.sum(x, axis=2))(attn_final_sent)
        output_list.append(attn_final_sent)

    word_attn = Reshape((sent_length*word_length, 256))(si_vects)
    x1 = Conv1D(256,2, activation='relu')(word_attn)
    x1_mp = GlobalMaxPooling1D()(x1)
    x1_av = GlobalAveragePooling1D()(x1)
    x2 = Conv1D(256,3, activation='relu')(word_attn)
    x2_mp = GlobalMaxPooling1D()(x2)
    x2_av = GlobalAveragePooling1D()(x2)
    x3 = Conv1D(256,4, activation='relu')(word_attn)
    x3_mp = GlobalMaxPooling1D()(x3)
    x3_av = GlobalAveragePooling1D()(x3)
    #x = Concatenate()([Flatten()(x1_mp), Flatten()(x2_mp),Flatten()(x3_mp)])
    #x = Concatenate()([x1_mp, x2_mp , x3_av])
    x = Maximum()([x1_mp, x1_av, x2_mp, x2_av , x3_mp, x3_av])
    x = BatchNormalization()(x)
    output_list.append(x)
    #x = Dense(256, activation='relu')(x)
    #x = Dropout(0.25)(x)
    #x = Dense(128, activation='relu')(x)
    #x = Dropout(0.25)(x)
    x = Multiply()(output_list)
    p = Dense(n_classes, activation='sigmoid')(x)

    model = Model(input, p)

    return model


def get_sentence_attention_combined_output(word_model , word_length, sent_length, n_classes):
    #x = Permute((2,1))(si_vects)
    nclasses = n_classes
    input = Input(shape=(sent_length, word_length ), dtype='int32')
    print(' input to sentence attn network',word_model)
    attentions_pred = []
    #print(output.summary())
    si_vects = TimeDistributed(word_model)(input)
    print('Shape after si_vects', si_vects.shape)
    u_it = TimeDistributed(TimeDistributed(Dense(100, activation='tanh')))(si_vects)
    print('Shape after word vector',u_it.shape)
    #h_it = TimeDistributed(Reshape((100,word_length)))(si_vects)
    #print('Shape after reshape word vector',h_it.shape)

    attn_final_word = [TimeDistributed(ATTNWORD(1))(u_it) for i in range(nclasses)]
    #a_it = Reshape(( word_length, 1))(a_it)
    #h_it = Reshape((word_length, 512))(h_it)
    print('ATTN Shape', attn_final_word[0].shape)
    attn_final_word = [Multiply()([si_vects, attn_final_word[i]]) for i in range(nclasses)]#Multiply()([h_it,a_it])
    print('Multi word Shape', attn_final_word[0].shape)
    attn_final_word = [Reshape((sent_length, 100,word_length))(attn_final_word[i]) for i in range(nclasses)]
    print ('Shape of the att1 is {}'.format(attn_final_word[0].shape))
    attn_final_word = [Lambda(lambda x: K.sum(x, axis=3))(attn_final_word[i]) for i in range(nclasses)]
    print ('Shape of the lambda word is {}'.format(attn_final_word[0].shape))
    attn_sents_for_all_classes = []
    for i in range(nclasses):
        x = Bidirectional(GRU(50,return_sequences=True))(attn_final_word[i])
        #x = Bidirectional(LSTM(256,return_sequences=True))(x)
        print('Shape after BD LSTM',x.shape)
        #x1 = Permute((2,1))(x)
        #print('Shape after permute',x1.shape)
        u_it = TimeDistributed(Dense(100, activation='tanh'))(x)
        print('Shape after word vector',u_it.shape)
        #h_it = Reshape((100,sent_length))(x)
        attn_final_sent = ATTNWORD(1)(u_it)
        print ('Shape of the sent att is {}'.format(attn_final_sent.shape))
        #attentions_pred.append(attn_final)
        attn_final_sent = Multiply()([x, attn_final_sent])
        print ('Shape of the multi sent att is {}'.format(attn_final_sent.shape))
        attn_final_sent = Reshape((100,sent_length))(attn_final_sent)
        attn_final_sent = Lambda(lambda x: K.sum(x, axis=2))(attn_final_sent)
        print ('Shape of the lambda sent att is {}'.format(attn_final_sent.shape))
        attn_sents_for_all_classes.append(attn_final_sent)
    x = Concatenate()(attn_sents_for_all_classes)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    #x = Dense(128, activation='relu')(x)
    #x = Dropout(0.2)(x)
    #x = Dense(64, activation='relu')(x)
    #x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    preds = Dense(nclasses, activation='sigmoid')(x)

    model = Model(input, preds)

    return model


"""
This returns LSTM based model. There are 6 output classes, all soft sharing the parameters of a common network.
"""
def get_model_soft_sharing_lstm_singleoutput(emb_matrix, sentence_length, word_length, learning_rate=0.001, n_classes=1, decay=0.1, combined_model=False):

    rmsprop = optimizers.Adam(lr=learning_rate, clipnorm=0.1, clipvalue=0.05,decay=decay)#
    word_model = get_word_attention(emb_matrix, word_length, rmsprop, n_classes)
    if not combined_model:
        model = get_sentence_attention(word_model, word_length, sentence_length, n_classes)
    else:
        model = get_sentence_attention_combined_output(word_model, word_length, sentence_length, n_classes)
    #model = attention_words_only(emb_matrix, word_length, 1)#sent_model
    #model.add(Activation('softmax'))
    #adam = optimizers.Adam(clipnorm=0.1,lr=learning_rate, clipvalue=0.05, decay=0.1)
    model.compile(loss='binary_crossentropy', optimizer=rmsprop,metrics=['accuracy'])
    #model.compile(loss='mse', optimizer=adam,metrics=['accuracy'])

    print (model.summary())

    return model


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))


lookup_words = tu.get_word_reverse_lookup(tokenizer)
comment_list = train.astype(str)[comment_col].tolist()
xtrain = tu.pad_sentences_sent(comment_list,SENTENCE,WORDS, tokenizer)

test_comment_list = test.astype(str)[comment_col].tolist()
xval = tu.pad_sentences_sent(test_comment_list,SENTENCE,WORDS, tokenizer)
print (xval.shape)

# Callbacks are passed to the model fit the `callbacks` argument in `fit`,
# which takes a list of callbacks. You can pass any number of callbacks.
callbacks_list = [
    # This callback will interrupt training when we have stopped improving
    keras.callbacks.EarlyStopping(
        # This callback will monitor the validation accuracy of the model
        monitor='val_loss',
        # Training will be interrupted when the accuracy
        # has stopped improving for *more* than 1 epochs (i.e. 2 epochs)
        patience=20,
    ),
    # This callback will save the current weights after every epoch
    keras.callbacks.ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,  # Path to the destination model file
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
), RocAucEvaluation(validation_data=(xval, YVal), interval=1)

#,  RocAucEvaluation(
#            validation_data = (xval, YVal)
#)#,

    #keras.callbacks.TensorBoard(
        # Log files will be written at this location
        #log_dir=TENSORFLOW_LOGDIR,
        # We will record activation histograms every 1 epoch
        #histogram_freq=1

#)


]


model = get_model_soft_sharing_lstm_singleoutput(final_emb_matrix, SENTENCE, WORDS, learning_rate=LEARNING_RATE, n_classes=6, decay=DECAY, combined_model=False)
#plot_model(model,to_file='han_attn_model_with_cnn.png')

tu.sentence_tokenizer("This is a great day!", 25)



def get_label_stat(y):
    #y = y.tolist()
    total_count = pd.Series(y).count()
    y1 = (pd.Series(y).sum()/total_count)*100
    y0 = 100-y1
    return total_count, y1, y0




total_count_train, y1, y0 = get_label_stat(YTrain[0])
print ('Training State - Total Records: {0}, Toxic percent: {1}, Normal percent: {2}'.format(total_count_train, y1, y0))
total_count_val, y1, y0 = get_label_stat(YVal[0])
print ('Validation State - Total Records: {0}, Toxic percent: {1}, Normal percent: {2}'.format(total_count_val, y1, y0))



model.fit(xtrain,YTrain ,batch_size=BATCH_SIZE, epochs=EPOCH_SIZE, verbose=1, validation_data=(xval, YVal), shuffle=True, callbacks=callbacks_list)#, callbacks=callbacks_list




test_df = pd.read_csv(TEST_FILE)
test_df = tu.clean_up(test_df)
#test_df['comment_text'] = test_df['comment_text'].apply(lambda x: tu.replace_unknown_words_with_UNK(x, tokenizer))
test_comments = test_df.astype(str)['comment_text'].tolist()
xtrain = tu.pad_sentences_sent(test_comments,SENTENCE,WORDS, tokenizer)
test_df.head()


predictions = model.predict(xtrain)


predicted_df = pd.DataFrame(columns=['id','toxic','severe_toxic','obscene','threat','insult','identity_hate'])
predicted_df['id'] = test_df['id']
for i, k in enumerate(pred_cols):
    predicted_df[k] = predictions[i]
predicted_df.head()


predicted_df.to_csv(OUTPUT_FILENAME,index=False, header=True)
