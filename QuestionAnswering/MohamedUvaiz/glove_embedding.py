import tensorflow as tf
import numpy as np
from collections import OrderedDict
import pandas as pd

file="glove.6B.50d.txt"

df=pd.read_csv(file,sep=" ",quoting=3, header=None, index_col=0)
glove = {key: val.values for key, val in df.T.items()}

words=list(glove.keys())
emb=np.array(list(glove.values()))

input_str = "like the country"
word_to_idx = OrderedDict({w:words.index(w) for w in input_str.split() if w in words})

tf.InteractiveSession()
tf_embedding = tf.constant(emb, dtype=tf.float32)
tf.nn.embedding_lookup(tf_embedding, list(word_to_idx.values())).eval()


