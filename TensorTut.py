from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# Data sets
IRIS_TRAINING = "winequality-red.csv"
IRIS_TEST = "testdata.csv"

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TRAINING,
                                                       target_dtype=np.float, target_column=-1, has_header=True)
test_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TEST,
                                                   target_dtype=np.float)

# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=11)]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.LinearRegressor(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10])

# Fit model
#classifier.fit(x=training_set.data, y=training_set.target, steps=2000)

# Input builders
def input_fn_train: # returns x, y
  ...
def input_fn_eval: # returns x, y
  ...

classifier.fit(input_fn=input_fn_train)
classifier.evaluate(input_fn=input_fn_eval)
classifier.predict(x=x)

