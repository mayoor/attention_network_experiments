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
import tokenizer_util as tu
import os.path
from keras.callbacks import Callback
from keras import backend as K

TRAIN_FILE_PATH = 'train.csv'#'/data/train.csv'
TEST_FILE = 'test.csv'#'/data/test.csv'

TIME_STEPS = 300
BATCH_SIZE = 256
LEARNING_RATE = 0.01 
DECAY = 0.25
EPOCH_SIZE = 10

TOKENIZER_FILE = 'tokenizer'
EMBEDDING_FILE = 'embedding'

TENSORFLOW_LOGDIR = 'logs'#'/output/tensorboard_logs'
MODEL_SAVE_PATH = 'models/best_model_new.h5' #'/output/best_model.h5'
OUTPUT_FILENAME = 'first_submission.csv'

def main():
	df = pd.read_csv(TRAIN_FILE_PATH)

	pred_cols = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

	df['total_classes'] = df['toxic']+df['severe_toxic']+df['obscene']+df['threat']+df['insult']+df['identity_hate']

	df = tu.clean_up(df)

	comment_list = df['comment_text'].tolist()
	max_len = TIME_STEPS
	comment_list.append("unk")
	n_classes = 1
	tokenizer = joblib.load(TOKENIZER_FILE)
	final_emb_matrix = joblib.load(EMBEDDING_FILE)



	class_count = []
	for col in pred_cols:
		class_count.append((col,len(df[df[col]==1])))
	print (class_count)

	train, test = train_test_split(df, test_size=0.10, random_state=42)
	train.head()



	XTrain = tokenizer.texts_to_sequences(train.astype(str)['comment_text'].tolist())
	XVal = tokenizer.texts_to_sequences(test.astype(str)['comment_text'].tolist())
	YTrain = np.array(train[['toxic','severe_toxic','obscene','threat','insult','identity_hate']])
	YVal = np.array(test[['toxic','severe_toxic','obscene','threat','insult','identity_hate']])

	train.tail()

	model = get_model_soft_sharing_lstm_singleoutput(final_emb_matrix, TIME_STEPS, learning_rate=LEARNING_RATE, n_classes=6, decay=DECAY)
	if os.path.isfile(MODEL_SAVE_PATH):
		print("Loading weights from existing path: {0}".format(MODEL_SAVE_PATH))
		model.load_weights(MODEL_SAVE_PATH)

	callbacks_list = define_callbacks()
	model.fit(pad_sequences(XTrain, TIME_STEPS),YTrain ,batch_size=BATCH_SIZE, epochs=EPOCH_SIZE, verbose=1, validation_data=(pad_sequences(XVal, TIME_STEPS), YVal), callbacks=callbacks_list)


	model.load_weights(MODEL_SAVE_PATH)
	model.evaluate(pad_sequences(XVal, TIME_STEPS), YVal, batch_size=BATCH_SIZE)

	test_df = pd.read_csv(TEST_FILE)
	test_df = tu.clean_up(test_df)
	test_comments = test_df['comment_text'].astype(str).tolist()
	XTest = tokenizer.texts_to_sequences(test_comments)
	print (test_df.columns)
	test_df.head()

	predictions = model.predict(pad_sequences(XTest, TIME_STEPS))


	predicted_df = pd.DataFrame(columns=['id','toxic','severe_toxic','obscene','threat','insult','identity_hate'])
	predicted_df['id'] = test_df['id']
	for i, k in enumerate(pred_cols):
		predicted_df[k] = predictions[:,i]
	predicted_df.head()


	predicted_df.to_csv(OUTPUT_FILENAME,index=False, header=True)



def get_stratified_train(df):
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
    return X_train, X_test


def rand_over_sample(number_of_records, records):
    sample = records.sample(n=number_of_records, replace=True)
    return sample


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


"""
This returns LSTM based model. There are 6 output classes, all soft sharing the parameters of a common network.
"""
def get_model_soft_sharing_lstm(emb_matrix, learning_rate=0.001, decay=.01):
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



"""
This returns LSTM based model. There are 6 output classes, all soft sharing the parameters of a common network.
"""
def get_model_soft_sharing_lstm_singleoutput(emb_matrix, time_steps, learning_rate=0.001, n_classes=1, decay=0.1):
    input = Input(shape=(time_steps,), dtype='int32')
    embedding = Embedding( input_dim=emb_matrix.shape[0], output_dim=emb_matrix.shape[1], weights=[emb_matrix],input_length=time_steps,trainable=True)

    sequence_input = embedding(input)
    x = Bidirectional(LSTM(128,return_sequences=True, dropout=0.2))(sequence_input)
    x = Bidirectional(LSTM(128,return_sequences=False, dropout=0.2))(x)
    x = Dropout(0.2)(x)

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
    model = Model(input,preds)
    adam = optimizers.Adam(lr=learning_rate, decay=DECAY)
    model.compile(loss='binary_crossentropy', optimizer=adam,metrics=['accuracy'])

    print (model.summary())

    return model

class LR(Callback):
	def on_epoch_begin(self, epoch, logs=None):
		print('The current LR is : {0}'.format(float(K.get_value(self.model.optimizer.lr))))
    	
def define_callbacks(monitor='val_loss', patience=10, model_path=MODEL_SAVE_PATH,logdir=TENSORFLOW_LOGDIR ):

	
	# Callbacks are passed to the model fit the `callbacks` argument in `fit`,
	# which takes a list of callbacks. You can pass any number of callbacks.
	callbacks_list = [
		# This callback will interrupt training when we have stopped improving
		keras.callbacks.EarlyStopping(
			# This callback will monitor the validation accuracy of the model
			monitor=monitor,
			# Training will be interrupted when the accuracy
			# has stopped improving for *more* than 1 epochs (i.e. 2 epochs)
			patience=patience,
		),
		# This callback will save the current weights after every epoch
		keras.callbacks.ModelCheckpoint(
			filepath=model_path,  # Path to the destination model file
			# The two arguments below mean that we will not overwrite the
			# model file unless `val_loss` has improved, which
			# allows us to keep the best model every seen during training.
			monitor='val_loss',
			save_best_only=True,
		),
	
		   keras.callbacks.ReduceLROnPlateau(
			   # This callback will monitor the validation loss of the model
			   monitor=monitor,
			   # It will divide the learning by 10 when it gets triggered
			   factor=0.1,
			   # It will get triggered after the validation loss has stopped improving
			   # for at least 10 epochs
			   patience=3,
	) ,

		keras.callbacks.TensorBoard(
			# Log files will be written at this location
			log_dir=logdir,
			# We will record activation histograms every 1 epoch
			histogram_freq=1,
			# We will record embedding data every 1 epoch
			embeddings_freq=1,
	) ,
	
		LR()

	]
	return callbacks_list


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


if __name__ == "__main__":
	main()