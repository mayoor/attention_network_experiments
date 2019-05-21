from keras.layers import LSTM, Bidirectional, Embedding, Input, Flatten, Dense, BatchNormalization, Dropout, Conv1D, Concatenate, MaxPool1D, AveragePooling1D, GlobalAveragePooling1D, GlobalMaxPool1D, SpatialDropout1D, TimeDistributed, Lambda
from keras.layers import Add
import keras
import tensorflow as tf
import keras.backend as K
from keras.models import Model
import numpy as np
import math

"""
Scaled Dot-Product Attention
"""
class SDPA(keras.layers.Layer):
    def __init__(self,output_dim, **kwargs):
        self.output_dim = output_dim
        super(SDPA,self).__init__(**kwargs)
    
    def build(self,input_shape):
        self.kernel_Q = self.add_weight(name='Q',
                                        shape=(input_shape[-1],self.output_dim),
                                        initializer='uniform',
                                        trainable=True
                                       )
        self.kernel_K = self.add_weight(name='K',
                                        shape=(input_shape[-1],self.output_dim),
                                        initializer='uniform',
                                        trainable=True
                                       )
        self.kernel_V = self.add_weight(name='V',
                                        shape=(input_shape[-1],self.output_dim),
                                        initializer='uniform',
                                        trainable=True
                                       )
    def call(self, x, mask=None):
        qvec = K.dot(x,self.kernel_Q)
        kvec = K.dot(x,self.kernel_K)
        vvec = K.dot(x,self.kernel_V)
#         print ("QVEC",qvec.shape)
#         print ("KVEC",kvec.shape)
        ktvec = K.permute_dimensions(kvec,(0,2,1))
#         print ("KTVEC",ktvec.shape)
        numerator = tf.einsum('ijk,ikz ->ijz',qvec,ktvec)
        smax = K.softmax((numerator/np.sqrt(self.output_dim)))
        final = tf.einsum('ijk,ikz ->ijz',smax,vvec)
        return final
    
    def compute_output_shape(self,input_shape):
        return(input_shape[0],input_shape[1],self.output_dim)
"""
Positional Encoder
"""
class PositionalEncoder(keras.layers.Layer):
    def __init__(self, **kwargs):
#         self.output_dim = output_dim
        super(PositionalEncoder,self).__init__(**kwargs)
     
    def build(self,input_shape):
        self.output_dim = input_shape[-1]
        def positional_encoding(shape, dtype=None):
            print ("PE shape is {}".format(shape))
            start = 0
            end = shape[1] // 2
            values = []
            for row in range(shape[0]):
                left = [math.sin(row/10000**((2*i)/shape[1])) for i in range(start,end)]
                right = [math.cos(row/10000**((2*i)/shape[1])) for i in range(start,end)]
                values.append(left+right)
            return np.array(values)
        self.kernel_pe = self.add_weight(name='pe',
                                        shape=(input_shape[-2],self.output_dim),
                                        initializer=positional_encoding,
                                        trainable=False
                                       )

    def call(self, x, mask=None):
        output = x+self.kernel_pe
        print ("The PE output shape is {}".format(output))
        return output
    
    def compute_output_shape(self,input_shape):
        return(input_shape[0],input_shape[1],self.output_dim)

def get_multi_head_attn(input, head_size, embedding_size,name="multi_head"):
    transformers = []
    for i in range(head_size):
        x1 = SDPA(100)(input)
        # xh = Dense(100, activation='relu')(x1)
        transformers.append(xh)
    x = Concatenate()(transformers)
    x = Dense(embedding_size, name=name)(x)
    return x

def get_transformer(input, head_size=5, embedding_size=100,index=0):
    multi_head = get_multi_head_attn(input, head_size, embedding_size, "multi_head_{}".format(index))
    add_out = Add()([multi_head, input])
    norm_out = LayerNormalization(name="multi_head_norm_{}".format(index))(add_out)

    ffout = TimeDistributed(Dense(embedding_size,activation='relu'))(norm_out)
    add_out = Add()([norm_out, ffout])
    norm_out = LayerNormalization(name="transformer_{}".format(index))(add_out)
    return norm_out

def get_transformer_model(multi_head_size,embedding,maxlen,depth=2):
    input = Input(shape=(maxlen,), name="input_sentence")
    x = Embedding(input_dim=embedding.shape[0], output_dim=embedding.shape[1], input_length=maxlen, weights=[embedding], trainable=False, mask_zero=False)(input)
    x = PositionalEncoder()(x)
    for i in range(depth):
        x = get_transformer(x,multi_head_size,embedding.shape[1],index=i)
    x = Flatten()(x)
    output = Dense(1, activation='sigmoid')(x)
    return Model(input,output)