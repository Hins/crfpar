# -*- coding: utf-8 -*-

from datetime import datetime
from parser import Model
from parser.utils.corpus import Corpus
from parser.utils.data import TextDataset, batchify
from parser.utils.field import Field
from pathlib import Path
from parser.cmds.cmd import CMD
import jieba
import json

import os
import time

import torch

class Predict(CMD):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Use a trained model to make predictions.'
        )
        subparser.add_argument('--marg', action='store_true',
                               help='whether to use marginal probs')
        subparser.add_argument('--proj', action='store_true',
                               help='whether to projectivise the outputs')
        subparser.add_argument('--fdata', default='data/ptb/test.conllx',
                               help='path to dataset')
        subparser.add_argument('--fpred', default='pred.conllx',
                               help='path to predicted result')
        subparser.add_argument('--input', default='',
                               help='path to dataset')
        subparser.add_argument('--result', default='',
                               help='path to predicted result')
        return subparser

    def __call__(self, args):
        super(Predict, self).__call__(args)
        print("Load the model")
        self.model = Model.load("./model/model")
        print(f"{self.model}\n")
        json_obj = []
        for file in Path(args.input).glob('**/*.txt'):
            print("file: {}".format(file))
            mid = os.path.join(args.input, "mid.json")
            print("mid: {}".format(mid))
            dst_json = os.path.join(args.result, str(file.stem) + '.json')
            print("dst_json: {}".format(dst_json))
            os.makedirs(os.path.dirname(dst_json), exist_ok=True)


            with open(str(file), 'r') as rf:
                sentences = []
                for line in rf:
                    sentence = [token for token in jieba.cut(line.replace("\n", ""))]
                    sentences.append(sentence)
            with open(mid, 'w') as f:
                for id, tokens in enumerate(sentences):
                    for idx, token in enumerate(tokens):
                        f.write(str(idx + 1) + "\t" + token + "\t" + token + "\t_\t_\t_\t_\t_\t_\t_\n")
                    f.write("\n")

            print("Load the dataset")
            self.fields = self.fields._replace(PHEAD=Field('probs'))
            corpus = Corpus.load(mid, self.fields)
            dataset = TextDataset(corpus, [self.WORD, self.FEAT], 5)
            os.remove(mid)
            # set the data loader
            dataset.loader = batchify(dataset, 32)
            print(f"{len(dataset)} sentences, "
                f"{len(dataset.loader)} batches")
            print("Make predictions on the dataset")
            test_start_time = time.time()
            pred_arcs, pred_rels, pred_probs = self.predict(dataset.loader)
            indices = torch.tensor([i for bucket in dataset.buckets.values()
                                    for i in bucket]).argsort()
            arcs_list = [pred_arcs[i] for i in indices]
            for i, sentence in enumerate(sentences):
                sentence_json_obj = {}
                sentence_json_obj["ID"] = i
                sentence_json_obj["text"] = "".join(sentence)
                sentence_json_obj["words"] = []
                for id, token in enumerate(sentence):
                    #if id == len(sentences):
                    #    break
                    token_obj = {}
                    token_obj["id"] = id + 1
                    token_obj["form"] = sentence[id]
                    token_obj["head"] = arcs_list[i][id]
                    token_obj["pos"] = ""
                    token_obj["deprel"] = ""
                    token_obj["stanfordnlpdependencies"] = ""
                    sentence_json_obj["words"].append(token_obj)
                json_obj.append(sentence_json_obj)
            test_time = time.time() - test_start_time
            with open(dst_json, 'w', encoding='utf-8') as f:
                json.dump(json_obj, f, indent=4, ensure_ascii=False)
        return len(json_obj), test_time, json_obj