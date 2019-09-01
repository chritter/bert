
# Copyright 2019 Google Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
from tensorflow import keras
import os
import re
import sys

import run_classifier
import optimization
import tokenization
import modeling

import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment('EventDetection')
from mlflow import tensorflow
tensorflow.autolog(every_n_iter=1) #default 100

# Parameters
# ==================================================
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("OUTPUT_DIR", 'models/model2',
                       """Path to output dir""")
tf.flags.DEFINE_string("train_data_file","data/aclImdb/aclImdb_train.csv",
                       """Path to training data. Expect 'text' and 'label' columns """)
tf.flags.DEFINE_string("test_data_file","data/aclImdb/aclImdb_test.csv",
                       """Path to development data """)
tf.flags.DEFINE_string("BERT_VOCAB","data/uncased_L-12_H-768_A-12/vocab.txt",
                       """Path to BERT vocab file """)
tf.flags.DEFINE_string("BERT_INIT_CHKPNT","data/uncased_L-12_H-768_A-12/bert_model.ckpt",
                       """Path to BERT model checkpoint """)
tf.flags.DEFINE_string("BERT_CONFIG","data/uncased_L-12_H-768_A-12/bert_config.json",
                       """Path to BERT model config file """)
tf.flags.DEFINE_integer("MAX_SEQ_LENGTH",128,
                       """max length of (token?) sequence. can increase up to 512 """)

# These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
tf.flags.DEFINE_integer("BATCH_SIZE",32,
                       """ batch size for training """)
tf.flags.DEFINE_float("LEARNING_RATE",2e-5,
                       """ learning rate for training """)
tf.flags.DEFINE_float("NUM_TRAIN_EPOCHS",3,
                       """ number of training epochs """)
# Warmup is a period of time where hte learning rate
# is small and gradually increases--usually helps training.
tf.flags.DEFINE_float("WARMUP_PROPORTION",0.1,
                       """ number of training epochs """)
tf.flags.DEFINE_integer("SAVE_CHECKPOINTS_STEPS",500,
                       """ number of checkpoints to save """)
tf.flags.DEFINE_integer("SAVE_SUMMARY_STEPS",100,
                       """ number of checkpoints to save """)
FLAGS = tf.flags.FLAGS
FLAGS(sys.argv, known_only=True)

# note that extra FLAGS paramters are saved in run_classifier.py. This were parameters set by Google.

# <--------- run specific settings

FLAGS.SAVE_CHECKPOINTS_STEPS=2
FLAGS.SAVE_SUMMARY_STEPS=1
FLAGS.OUTPUT_DIR = 'models/model5'
FLAGS.NUM_TRAIN_EPOCHS = 1




for key, values in FLAGS.flag_values_dict().items():
    mlflow.log_param(key,values)

# end of parameters


tf.gfile.MakeDirs(FLAGS.OUTPUT_DIR)

tf.logging.info('***** Model output directory: {} *****'.format(FLAGS.OUTPUT_DIR))

# <------------------ Load the data

# Load all files from a directory in a DataFrame.
train = pd.read_csv(FLAGS.train_data_file)
test = pd.read_csv(FLAGS.test_data_file)
train = train.sample(70)
test = test.sample(70)

# label_list is the list of labels, i.e. True, False or 0, 1 or 'dog', 'cat'
label_list = list(np.unique(train['label'])) #[0, 1]

tf.logging.info('shape of data: train: (%d,%d), test: (%d,%d)' % (train.shape+test.shape))
tf.logging.info('columns of train file: %s' % ','.join(train.columns))


# <------------------ Prepare the training input

tf.logging.info('prepare training input...')

# Use the InputExample class from BERT's run_classifier code to create examples from the data. Each data point
# wrapped into a InputExample class
train_InputExamples = train.apply(lambda x: run_classifier.InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this example
                                                                   text_a = x['text'],
                                                                   text_b = None, 
                                                                   label = x['label']), axis = 1)

test_InputExamples = test.apply(lambda x: run_classifier.InputExample(guid=None,
                                                                   text_a = x['text'],
                                                                   text_b = None, 
                                                                   label = x['label']), axis = 1)


# <------------------ Prepare tokenizer and do tokenization

tf.logging.info('prepare tokenizer')

# Checks whether the casing config is consistent with the checkpoint name.
tokenization.validate_case_matches_checkpoint(do_lower_case=True,init_checkpoint=FLAGS.BERT_INIT_CHKPNT)

# build the tokenizer
tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.BERT_VOCAB,do_lower_case=True)

tf.logging.info('created tokenizer')
#tokens_test = tokenizer.tokenize("This here's an example of using the BERT tokenizer")
#tf.logging.info('example tokenization:'+)



# Convert our train and test features to InputFeatures that BERT understands.
# creates lists with elements/sentences of type InputFeatures class
train_features = run_classifier.convert_examples_to_features(train_InputExamples, label_list, FLAGS.MAX_SEQ_LENGTH, tokenizer)
test_features = run_classifier.convert_examples_to_features(test_InputExamples, label_list, FLAGS.MAX_SEQ_LENGTH, tokenizer)


# <------------------ Prepare model


def create_model(bert_config,is_predicting, input_ids, input_mask, segment_ids, labels,
                 num_labels):

  # this initializes the BERT model, see model code in modeling module!
  model = modeling.BertModel(
    config=bert_config,
    is_training=not is_predicting,
    input_ids=input_ids,
    input_mask=input_mask,
    token_type_ids=segment_ids)# ,
    #use_one_hot_embeddings=use_one_hot_embeddings)

  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  # Create our own layer to tune for politeness data.
  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):

    # Dropout helps prevent overfitting
    output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    # Convert labels into one-hot encoding
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
    # If we're predicting, we want predicted labels and the probabiltiies.
    if is_predicting:
      return (predicted_labels, log_probs)

    # If we're train/eval, compute loss between predicted and actual label
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, predicted_labels, log_probs)


# model_fn_builder actually creates our model function
# using the passed parameters for num_labels, learning_rate, etc.
def model_fn_builder(bert_config,init_checkpoint,num_labels, learning_rate, num_train_steps,
                     num_warmup_steps):
  """Returns `model_fn` closure for TPUEstimator."""
  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]

    is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)
    
    # TRAIN and EVAL
    if not is_predicting:

      # create the model, specifically for training
      (loss, predicted_labels, log_probs) = create_model(bert_config,
        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

      if mode == tf.estimator.ModeKeys.TRAIN:
          # init weights (added CR)
          tvars = tf.trainable_variables()
          initialized_variable_names = {}
          if init_checkpoint:
              tf.logging.info('start loading weights %d from checkpoint %s' % (len(tvars),init_checkpoint))
              (assignment_map, initialized_variable_names
               ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
              # load weight maps
              tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
              tf.logging.info('loading weights done')
              tf.logging.info("**** Trainable Variables ****")


      # creates optimizer operation, based on Adam and exponential decaying lr
      train_op = optimization.create_optimizer(
          loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

      # Calculate evaluation metrics.
      accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
      def metric_fn(label_ids, predicted_labels):
        f1_score = tf.contrib.metrics.f1_score(
            label_ids,
            predicted_labels)
        #mlflow.log_metric('f1_score test',f1_score[1])
        auc = tf.metrics.auc(
            label_ids,
            predicted_labels)
        recall = tf.metrics.recall(
            label_ids,
            predicted_labels)
        precision = tf.metrics.precision(
            label_ids,
            predicted_labels) 
        true_pos = tf.metrics.true_positives(
            label_ids,
            predicted_labels)
        true_neg = tf.metrics.true_negatives(
            label_ids,
            predicted_labels)   
        false_pos = tf.metrics.false_positives(
            label_ids,
            predicted_labels)  
        false_neg = tf.metrics.false_negatives(
            label_ids,
            predicted_labels)
        return {
            #"eval_accuracy": accuracy,
            "f1_score": f1_score,
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "true_positives": true_pos,
            "true_negatives": true_neg,
            "false_positives": false_pos,
            "false_negatives": false_neg
        }

      eval_metrics = metric_fn(label_ids, predicted_labels)
      #mlflow.log_metric('eval_metrics recall',eval_metrics['recall'])
      # this metric will be logging only during evaluation
      eval_metrics['eval_accuracy'] = accuracy

      # this metric will be logging only during training
      tf.summary.scalar('accuracy', accuracy[1])

      # two mode optoins: TRAIN or EVAL, train here
      if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode,
          loss=loss,
          train_op=train_op)
          #eval_metric_ops=eval_metrics)
      else:
          # eval metrics are being printed to tf log output and saved to disk
          return tf.estimator.EstimatorSpec(mode=mode,
            loss=loss,
            eval_metric_ops=eval_metrics)
    else:



      # create the model specifically for predictions
      (predicted_labels, log_probs) = create_model(bert_config,
        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

      predictions = {
          'probabilities': log_probs,
          'labels': predicted_labels
      }
      return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  # Return the actual model function in the closure
  return model_fn


# Compute train and warmup steps from batch size

# Compute # train and warmup steps from batch size
num_train_steps = int(len(train_features) / FLAGS.BATCH_SIZE * FLAGS.NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * FLAGS.WARMUP_PROPORTION)

tf.logging.info('num warmump steps: %d' % (num_warmup_steps))
tf.logging.info('num training steps: %d' % (num_train_steps))


# specifies the configurations for an Estimator run.
# Specify outpit directory and number of checkpoint steps to save
# model_dir: directory where model parameters, graph, etc are saved.
# save_summary_steps: Save summaries every this many steps.
run_config = tf.estimator.RunConfig(
    model_dir=FLAGS.OUTPUT_DIR,
    save_summary_steps=FLAGS.SAVE_SUMMARY_STEPS,
    save_checkpoints_steps=FLAGS.SAVE_CHECKPOINTS_STEPS)

# build the model
bert_config = modeling.BertConfig.from_json_file(FLAGS.BERT_CONFIG)

model_fn = model_fn_builder(
  bert_config=bert_config,
  init_checkpoint = FLAGS.BERT_INIT_CHKPNT,
  num_labels=len(label_list),
  learning_rate=FLAGS.LEARNING_RATE,
  num_train_steps=num_train_steps,
  num_warmup_steps=num_warmup_steps)

# initializes the estimator
estimator = tf.estimator.Estimator(
  model_fn=model_fn,
  config=run_config,
  params={"batch_size": FLAGS.BATCH_SIZE})



# <------------------ Begin training

# Create an input function for training. drop_remainder = True for using TPUs.
train_input_fn = run_classifier.input_fn_builder(
    features=train_features,
    seq_length=FLAGS.MAX_SEQ_LENGTH,
    is_training=True,
    drop_remainder=False)

tf.logging.info('Beginning Training!')
current_time = datetime.now()
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
tf.logging.info("Training took time %s" % (datetime.now() - current_time))


# <------------------ Begin evaluation on test data

tf.logging.info('Beginning Evaluation!')
test_input_fn = run_classifier.input_fn_builder(
    features=test_features,
    seq_length=FLAGS.MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=False)

current_time = datetime.now()
estimator.evaluate(input_fn=test_input_fn, steps=None)
tf.logging.info("evaluation took time %s" % (datetime.now() - current_time))



# <------------------ Prediction: Apply prediction function to text data


def getPrediction(in_sentences):
  '''
  Function to provide class predictions for list of input sentences
  :param in_sentences:
  :return:
  '''

  #labels = ["Negative", "Positive"]

  # pre-process input
  input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = 0) for x in in_sentences] # here, "" is just a dummy label
  input_features = run_classifier.convert_examples_to_features(input_examples, label_list, FLAGS.MAX_SEQ_LENGTH, tokenizer)

  # create model function input
  predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=FLAGS.MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)

  # calculate the predictions
  predictions = estimator.predict(predict_input_fn)

  #
  output =  [(sentence, prediction['probabilities'], prediction['labels']) for sentence, prediction in zip(in_sentences, predictions)]
  return output

tf.logging.info("start prediction")

# input test sentences
pred_sentences = [
  "That movie was absolutely awful",
  "The acting was a bit lacking",
  "The film was creative and surprising",
  "Absolutely fantastic!"
]

# get predictions for sentences
predictions = getPrediction(pred_sentences)

print(predictions)
