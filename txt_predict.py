#! /usr/bin/env python
# encoding=utf-8

"""
Batch Prediction
"""

import codecs
from datetime import datetime
from functools import reduce
import logging
from operator import iconcat
import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf

from .bert_lstm_ner import create_model, InputFeatures
from .bert import tokenization, modeling
#np.set_printoptions(threshold=np.inf)

class ChineseNER:
    def __init__(self, model_dir, batch_size=1, max_seq_length=128):

        self.batch_size = batch_size

        self.max_seq_length = max_seq_length

        self.do_lower_case = True

        script_dir = os.path.dirname(os.path.abspath(__file__))

        if model_dir is None:
            self.model_dir = os.path.join(script_dir, 'output')
        else:
            self.model_dir = model_dir

        self.bert_path = os.path.join(script_dir, 'chinese_L-12_H-768_A-12')

        logging.info('Checkpoint path: %s', os.path.join(self.model_dir, "checkpoint"))
        if not os.path.exists(os.path.join(self.model_dir, "checkpoint")):
            raise Exception("failed to get checkpoint. going to return ")

        with codecs.open(os.path.join(self.model_dir, 'label2id.pkl'), 'rb') as rf:
            label2id = pickle.load(rf)
            self.id2label = {value: key for key, value in label2id.items()}

        with codecs.open(os.path.join(self.model_dir, 'label_list.pkl'), 'rb') as rf:
            self.label_list = pickle.load(rf)
        num_labels = len(self.label_list) + 1

        is_training = False
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = False
        self.sess = tf.Session(config=gpu_config)

        self.tokenizer = tokenization.FullTokenizer(
                vocab_file=os.path.join(self.bert_path, 'vocab.txt'),
                do_lower_case=self.do_lower_case)

        global graph
        graph = tf.get_default_graph()
        with graph.as_default():
            logging.info("Going to restore checkpoint")
            #sess.run(tf.global_variables_initializer())

            self.input_ids_p = tf.placeholder(tf.int32, [None, self.max_seq_length],
                                              name="input_ids")
            self.label_ids_p = tf.placeholder(tf.int32, [None, self.max_seq_length],
                                              name="label_ids")
            self.input_mask_p = tf.placeholder(tf.int32, [None, self.max_seq_length],
                                               name="input_mask")

            bert_config = modeling.BertConfig.from_json_file(
                os.path.join(self.bert_path, 'bert_config.json'))
            (total_loss, logits, trans, self.pred_ids) = create_model(
                bert_config,
                is_training,
                self.input_ids_p,
                self.input_mask_p,
                self.label_ids_p,
                num_labels)

            saver = tf.train.Saver()
            saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))

    def predict_sentences(self, sentences, as_data=False):

        def convert_single_example1(ex_index, example, label_list, max_seq_length, tokenizer, mode):
            #Converts a single `InputExample` into a single `InputFeatures`.
            label_map = {}

            for (i, label) in enumerate(label_list, 1):
                label_map[label] = i

            tokens = example

            if len(tokens) >= self.max_seq_length - 1:
                tokens = tokens[0:(self.max_seq_length - 2)]

            ntokens = []
            label_ids = []
            ntokens.append(tokenization.Token("[CLS]"))

            label_ids.append(label_map["[CLS]"])
            for i, token in enumerate(tokens):
                ntokens.append(token)
                label_ids.append(0)
            ntokens.append(tokenization.Token("[SEP]"))

            label_ids.append(label_map["[SEP]"])
            input_ids = tokenizer.convert_tokens_to_ids(ntokens)
            input_mask = [1] * len(input_ids)

            while len(input_ids) < self.max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                label_ids.append(0)
                ntokens.append(tokenization.Token("**NULL**"))

            assert len(input_ids) == self.max_seq_length
            assert len(input_mask) == self.max_seq_length
            assert len(label_ids) == self.max_seq_length

            feature = InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                label_ids=label_ids
            )
            return feature

        def convert_batch(lines):
            features = [convert_single_example1(0, line, self.label_list, self.max_seq_length,
                                                self.tokenizer, 'p')
                        for line in lines]

            input_ids = []
            input_mask = []
            label_ids = []

            for feature in features:
                input_ids.append(feature.input_ids)
                input_mask.append(feature.input_mask)
                label_ids.append(feature.label_ids)

            return input_ids, input_mask, label_ids, len(features)

        def convert_id_to_label(pred_ids_result, idx2label, batch_size):
            result = []
            logging.debug('pred_ids_result: %d %s', len(pred_ids_result), type(pred_ids_result))
            pred_ids_result = reduce(iconcat, pred_ids_result, [])

            assert batch_size == len(pred_ids_result)
            for results in pred_ids_result:
                curr_seq = []
                for ids in results:
                    if ids == 0:
                        break
                    curr_label = idx2label[ids]
                    if curr_label in ['[CLS]', '[SEP]']:
                        continue
                    curr_seq.append(curr_label)
                result.append(curr_seq)
            return result

        global graph
        with graph.as_default():
            logging.debug('id2label: %s', self.id2label)

            data = []
            allTokens = []
            start = datetime.now()
            for ind, offset, sentence in \
                    zip(sentences.paragraphId, sentences.offset, sentences.sentence):
                logging.debug("sentence %d: %s", ind, sentence)
                tokens = self.tokenizer.tokenize_cleaned(str(sentence))
                #logging.debug('%s', tokens)
                allTokens.append(tokens)

            input_ids, input_mask, label_ids, batch_size = convert_batch(allTokens)
            feed_dict = {self.input_ids_p: input_ids,
                        self.input_mask_p: input_mask,
                        self.label_ids_p: label_ids}
            pred_ids_result = self.sess.run([self.pred_ids], feed_dict)
            pred_label_result = convert_id_to_label(pred_ids_result, self.id2label, batch_size)

            logging.debug(pred_label_result)

            for i in range(len(allTokens)):
                label_start = -1
                label_type = ''
                last_token = None

                for label, token in zip(pred_label_result[i], allTokens[i]):

                    if not label.startswith('I-'):
                        if label_start >= 0:
                            data.append([
                                sentences.paragraphId[i],
                                sentences.offset[i],
                                label_type,
                                sentences.sentence[i],
                                label_start,
                                last_token.offset + last_token.original_length
                            ])

                        if label.startswith('B-'):
                            label_start = token.offset
                            label_type = label[2:]
                        elif label == 'O':
                            label_start = -1
                        else:
                            raise Exception('Unrecognized label {} for sentence {} '
                                            'and token {}' \
                                            .format(label, sentences.sentence[i], token))
                    last_token = token

                if label_start >= 0:
                    data.append([sentences.paragraphId[i], sentences.offset[i], label_type, sentences.sentence[i], label_start, len(sentences.sentence[i])])

            #strage_combined_link_org_loc(sentence, pred_label_result[0])
            logging.debug('time used: %d sec', (datetime.now() - start).total_seconds())
            if as_data:
                return data
            return pd.DataFrame(data=data, columns=['Label', 'Entity', 'Context', 'StartPos'])

class Pair:
    def __init__(self, word, start, end, type, merge=False):
        self.__word = word
        self.__start = start
        self.__end = end
        self.__merge = merge
        self.__types = type

    @property
    def start(self):
        return self.__start
    @property
    def end(self):
        return self.__end
    @property
    def merge(self):
        return self.__merge
    @property
    def word(self):
        return self.__word

    @property
    def types(self):
        return self.__types
    @word.setter
    def word(self, word):
        self.__word = word
    @start.setter
    def start(self, start):
        self.__start = start
    @end.setter
    def end(self, end):
        self.__end = end
    @merge.setter
    def merge(self, merge):
        self.__merge = merge

    @types.setter
    def types(self, type):
        self.__types = type

    def __str__(self) -> str:
        line = []
        line.append('entity:{}'.format(self.__word))
        line.append('start:{}'.format(self.__start))
        line.append('end:{}'.format(self.__end))
        line.append('merge:{}'.format(self.__merge))
        line.append('types:{}'.format(self.__types))
        return '\t'.join(line)


class Result:
    def __init__(self, config):
        self.config = config
        self.person = []
        self.loc = []
        self.org = []
        self.others = []

    def get_result(self, tokens, tags):
        self.result_to_json(tokens, tags)
        return self.person, self.loc, self.org

    def result_to_json(self, string, tags):

        item = {"entities": []}
        entity_name = ""
        entity_start = 0
        idx = 0
        last_tag = ''

        for char, tag in zip(string, tags):
            if tag[0] == "S":
                self.append(char, idx, idx+1, tag[2:])
                item["entities"].append({"word": char, "start": idx, "end": idx+1, "type":tag[2:]})
            elif tag[0] == "B":
                if entity_name != '':
                    self.append(entity_name, entity_start, idx, last_tag[2:])
                    item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
                    entity_name = ""
                entity_name += char
                entity_start = idx
            elif tag[0] == "I":
                entity_name += char
            elif tag[0] == "O":
                if entity_name != '':
                    self.append(entity_name, entity_start, idx, last_tag[2:])
                    item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
                    entity_name = ""
            else:
                entity_name = ""
                entity_start = idx
            idx += 1
            last_tag = tag
        if entity_name != '':
            self.append(entity_name, entity_start, idx, last_tag[2:])
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
        return item

    def append(self, word, start, end, tag):
        if tag == 'Location':
            self.loc.append(Pair(word, start, end, 'LOC'))
        elif tag == 'Person':
            self.person.append(Pair(word, start, end, 'PER'))
        elif tag == 'Organization':
            self.org.append(Pair(word, start, end, 'ORG'))
        else:
            self.others.append(Pair(word, start, end, tag))
