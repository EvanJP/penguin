import tensorflow as tf
import numpy as np
import os
import dataclasses
from typing import List


class Time2VectorLayer(tf.keras.layers.Layer):
    """
    t2v(gamma)[i] = {w_i*gamma + phi_i} if i = 0.
    t2v(gamma)[i] = {F(w_i*gamma + phi_i)} if 1 <= i <= k
    F is the periodic function.

    Take in (batch_size, seq_len, num_features).
    num_features = 5 for Open, High, Low, Close, Volume.
    """

    def __init__(self, seq_len):
        super(Time2VectorLayer, self).__init__()
        self.seq_len = seq_len

    def build(self, input_shape):
        self.linear = tf.keras.layers.Dense(self.seq_len)
        self.periodic = tf.keras.layers.Dense(self.seq_len)

    def call(self, inputs):
        # inputs: (batch_size, seq_len, num_features)
        x = tf.math.reduce_mean(
            inputs[:, :, : 4],
            axis=-1)
        time_linear = self.linear(x)
        time_linear = tf.expand_dims(time_linear, axis=-1)
        time_periodic = tf.math.sin(self.linear(x))
        time_periodic = tf.expand_dims(time_periodic, axis=-1)
        return tf.concat([time_linear, time_periodic], axis=-1)


class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_k, d_v):
        super(SelfAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v

    def build(self, input_shape):
        self.query = tf.keras.layers.Dense(self.d_k, input_shape=input_shape)
        self.key = tf.keras.layers.Dense(self.d_k, input_shape=input_shape)
        self.value = tf.keras.layers.Dense(self.d_v, input_shape=input_shape)

    def call(self, inputs):
        # inputs: (in_seq, in_seq, in_seq)
        q = self.query(inputs[0])
        k = self.key(inputs[1])
        v = self.value(inputs[2])

        attn_weights = tf.matmul(q, k, transpose_b=True)
        attn_weights = tf.map_fn(lambda x: x/np.sqrt(self.d_k), attn_weights)
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)
        return tf.matmul(attn_weights, v)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.attn_heads = []

    def build(self, input_shape):
        self.attn_heads = [SelfAttention(self.d_k, self.d_v)
                           for _ in range(self.n_heads)]
        self.linear = tf.keras.layers.Dense(
            input_shape[0][-1])

    def call(self, inputs):
        attn = [self.attn_heads[i](inputs) for i in range(self.n_heads)]
        concat_attn = tf.concat(attn, axis=-1)
        return self.linear(concat_attn)


class TransformerEncoderBlock(tf.keras.Model):
    def __init__(self, d_k, d_v, n_heads, ff_dim, out_dim, dropout=0.1, **kwargs):
        super(TransformerEncoderBlock, self).__init__(name='')
        self.attn_multi = MultiHeadAttention(d_k, d_v, n_heads)
        self.attn_dropout = tf.keras.layers.Dropout(dropout)
        self.attn_normalize = tf.keras.layers.LayerNormalization()

        self.ff = tf.keras.layers.Dense(ff_dim, activation='relu')
        self.proj = tf.keras.layers.Dense(out_dim)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.normalize = tf.keras.layers.LayerNormalization()

    def call(self, inputs, training=False):
        x = self.attn_multi(inputs)
        x = self.attn_dropout(x, training=training)
        x = self.attn_normalize(inputs[0] + x)
        x = self.ff(x)
        x = self.proj(x)
        x = self.dropout(x, training=training)
        return self.normalize(inputs[0] + x)


def create_time_2_vector_transformer(
        seq_len, d_k, d_v, n_heads, ff_dim, out_dim,
        num_features, proj_dim, dropout=0.1):
    time_embedding = Time2VectorLayer(seq_len)
    transformer_1 = TransformerEncoderBlock(
        d_k, d_v, n_heads, ff_dim, out_dim, dropout)
    transformer_2 = TransformerEncoderBlock(
        d_k, d_v, n_heads, ff_dim, out_dim, dropout)
    inputs = tf.keras.Input(shape=(seq_len, num_features))
    x = time_embedding(inputs)
    x = tf.keras.layers.Concatenate()([inputs, x])
    x = transformer_1((x, x, x))
    x = transformer_2((x, x, x))
    x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(proj_dim, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(1, activation='linear')(x)

    model = tf.keras.Model(inputs=inputs, outputs=out)
    # mean square error, mean absolute error, and mean absolute percentage error
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])
    return model
