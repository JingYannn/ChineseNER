#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import configparser
import logging
import os

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers

from .bert import modeling
from .bert import optimization, tokenization
from .lstm_crf_layer import BLSTM_CRF

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, label_ids, ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_ids = label_ids

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data."""
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                tokens = contends.split(' ')
                if len(tokens) == 2:
                    word = line.strip().split(' ')[0]
                    label = line.strip().split(' ')[-1]
                else:
                    if len(contends) == 0:# Next sequence
                        l = ' '.join([label for label in labels if len(label) > 0])
                        w = ' '.join([word for word in words if len(word) > 0])
                        lines.append([l, w])
                        words = []
                        labels = []
                        continue
                if contends.startswith("-DOCSTART-"):
                    words.append('')
                    continue
                words.append(word)
                labels.append(label)
            return lines

class NerProcessor(DataProcessor):
    """Processor for Ner task."""

    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        labels = ["O", "X", "[CLS]", "[SEP]"]

        script_dir = os.path.dirname(os.path.abspath(__file__))

        config = configparser.ConfigParser()

        config.read(os.path.join(script_dir, 'config.ini'), encoding="utf-8")

        label_list = config.get('ChineseNER', 'label_list').split(",")

        for label in label_list:
            labels.append("B-"+label)
            labels.append("I-"+label)

        return labels

        #return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]

        #return ['I-Location', 'B-Physical', 'I-Physical',
        #        'B-Thing', 'I-Term', 'I-Person', 'B-Location', 'B-Abstract', 'I-Abstract',
        #        'B-Metric', 'I-Thing', 'I-Metric', 'B-Organization', 'I-Time',
        #        'B-Person', 'B-Term', 'I-Organization', 'B-Time', '[CLS]', '[SEP]', 'X', 'O']


    def _create_example(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            if i == 0:
                print(label)
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples

def create_model(bert_config, is_training, input_ids, input_mask,
                 labels, num_labels):
    """Creates a classification model.."""
    # Load Bert model and get the corresponding word embedding

    script_dir = os.path.dirname(os.path.abspath(__file__))

    config = configparser.ConfigParser()

    config.read(os.path.join(script_dir, 'config.ini'), encoding="utf-8")

    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask)

    embedding = model.get_sequence_output()
    max_seq_length = embedding.shape[1].value
    used = tf.sign(tf.abs(input_ids))
    lengths = tf.reduce_sum(used, reduction_indices=1)
    blstm_crf = BLSTM_CRF(embedded_chars=embedding,
                          hidden_unit=config.getint('ChineseNER', 'lstm_size'),
                          cell_type=config.get('ChineseNER', 'cell'),
                          num_layers=config.getint('ChineseNER', 'num_layers'),
                          dropout_rate=config.getfloat('ChineseNER', 'dropout_rate'), initializers=initializers,
                          num_labels=num_labels, seq_length=max_seq_length, labels=labels,
                          lengths=lengths, is_training=is_training)
    rst = blstm_crf.add_blstm_crf_layer(crf_only=False)

    return rst

def get_last_checkpoint(model_path):
    if not os.path.exists(os.path.join(model_path, 'checkpoint')):
        logging.warning('checkpoint does not exist: %s', os.path.join(model_path, 'checkpoint'))
        return None
    last = None
    with codecs.open(os.path.join(model_path, 'checkpoint'), 'r', encoding='utf-8') as fd:
        for line in fd:
            line = line.strip().split(':')
            if len(line) != 2:
                continue
            if line[0] == 'model_checkpoint_path':
                last = line[1][2:-1]
                break
    return last


def adam_filter(model_path):
    """Remove the parameters of Adam in the model, which are useless in the prediction."""
    last_name = get_last_checkpoint(model_path)
    if last_name is None:
        return
    sess = tf.Session()
    imported_meta = tf.train.import_meta_graph(os.path.join(model_path, last_name + '.meta'))
    imported_meta.restore(sess, os.path.join(model_path, last_name))
    need_vars = []
    for var in tf.global_variables():
        if 'adam_v' not in var.name and 'adam_m' not in var.name:
            need_vars.append(var)
    saver = tf.train.Saver(need_vars)
    saver.save(sess, os.path.join(model_path, 'model.ckpt'))


if __name__ == "__main__":
    tf.app.run()
    # filter model
    script_dir = os.path.dirname(os.path.abspath(__file__))

    conf = configparser.ConfigParser()

    conf.read(os.path.join(script_dir, 'config.ini'), encoding="utf-8")

    output_dir = os.path.join(script_dir, 'output')

    if conf.getboolean('ChineseNER', 'filter_adam_var'):
        adam_filter(output_dir)
