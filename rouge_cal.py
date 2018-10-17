#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copytright (c) 2017 Baidu.com, Inc. All right reserved.
#
################################################################################
"""
Desc:
Usage:

 Author: Zebin Lin (linzebin@baidu.com)
 Create time: 2017-07-07
 Filename:rouge_cal.py
"""

import os
import sys
import logging
import json

from rouge import Rouge
# import tensorflow as tf
from collections import defaultdict

reload(sys)
sys.setdefaultencoding('utf-8')

# FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_string("ref_dir", '', 'reference summary data dir')
# tf.app.flags.DEFINE_string("decode_dir", '', 'decode summary data dir')

ref_dir = "ref"
decode_dir = "dec"


def output_result(scores):
    for standard in ['rouge-1', 'rouge-2', 'rouge-l']:
        print "STANDARD: %s" % standard
        print "=========="
        print "Precision: %s" % (scores[standard].get('p', 0.0))
        print "Recall: %s" % (scores[standard].get('r', 0.0))
        print "F1: %s" % (scores[standard].get('f', 0.0))
        print ""


def main():
    rouge = Rouge()
    decode_files = sorted(os.listdir(decode_dir))

    ref_sentences = []
    decode_sentences = []

    for _file in decode_files:
        ref_file = _file.split('_')[0] + '_reference.txt'
        ref_file = os.path.join(ref_dir, ref_file)
        if not os.path.exists(ref_file):
            logging.warning(
                'Miss reference summary: %s | %s' % (_file, ref_file))
            continue

        decode_file = os.path.join(decode_dir, _file)
        with open(decode_file, 'r') as fin:
            decode_line = ''
            for line in fin:
                decode_line += line
            decode_sentences.append(decode_line)

        with open(ref_file, 'r') as fin:
            ref_line = ''
            for line in fin:
                ref_line += line
            ref_sentences.append(ref_line)

    scores_list = []
    for i in range(len(decode_sentences)):
        try:
            scores = rouge.get_scores(
                decode_sentences[i], ref_sentences[i], avg=True)
            scores_list.append(scores)
        except Exception as e:
            logging.info('Get scores of rouge error: %s' % i)
            logging.error(e)
            continue

    final_scores = defaultdict(lambda: defaultdict(float))
    for standard in ['rouge-1', 'rouge-2', 'rouge-l']:
        precision = sum([score[standard]['p'] for score in scores_list
                        ]) / len(scores_list)
        final_scores[standard]['p'] = precision
        recall = sum([score[standard]['r'] for score in scores_list
                     ]) / len(scores_list)
        final_scores[standard]['r'] = recall
        f1 = sum([score[standard]['f'] for score in scores_list
                 ]) / len(scores_list)
        final_scores[standard]['f'] = f1

    output_result(final_scores)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=
        '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%d %b %Y %H:%M:%S')
    main()
