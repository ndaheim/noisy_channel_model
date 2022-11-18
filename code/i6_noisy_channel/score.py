import spacy
from dataset_walker import DatasetWalker
from collections import Counter

import bert_score
import numpy as np
import sacrebleu
from datasets import load_dataset
from nltk import ngrams
from nltk.tokenize import RegexpTokenizer
from rouge_score import rouge_scorer

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from rouge import Rouge

import re

import sys
import json
import argparse
from collections import defaultdict

RE_ART = re.compile(r'\b(a|an|the)\b')
RE_PUNC = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')
tokenizer = RegexpTokenizer(r'\w+')

nlp = spacy.load("en_core_web_sm")


def add_bertscore(pred, ref):
    P, R, F1 = bert_score.score(
        pred, ref, lang="en", verbose=False, rescale_with_baseline=True)
    return F1.detach().numpy()


def get_tokens(text):
    doc = nlp(text)
    tokens = [tok.text.lower()
              for tok in doc if not tok.is_stop and not tok.is_punct]
    return tokens


def ngram_stats(data, N):
    data = [tokenizer.tokenize(item) for item in data]
    data = [[token.lower() for token in item] for item in data]
    ngram_freqs = {}  # ngrams with frequencies
    ngram_len = 0  # total number of ngrams
    for item in data:
        for ngram in ngrams(item, N):
            ngram_freqs[ngram] = ngram_freqs.get(ngram, 0) + 1
            ngram_len += 1
    if N > 1:
        repetitions = sum([1 for value in ngram_freqs.values() if value > 2])
    else:
        repetitions = 0
    # number of unique ngrams
    unique_ngrams = len([val for val in ngram_freqs.values() if val == 1])
    return ngram_freqs, unique_ngrams, ngram_len, repetitions


def entropy(ngram_freqs) -> float:
    """Shannon entropy over ngram frequencies"""
    total_freq = sum(ngram_freqs.values())
    return -sum(
        [
            freq / total_freq * np.log2(freq / total_freq)
            for freq in ngram_freqs.values()
        ]
    )


def selection_range(upper):
    i = 1
    j = 0
    while i <= upper:
        yield i
        if j == 0:
            i *= 5
        else:
            i *= 2
        j = (j + 1) % 2


def f1_score(gold, pred):
    gold_toks = get_tokens(gold)
    pred_toks = get_tokens(pred)

    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


class Metric:
    def __init__(self, selection_topks=None, knowledge=None):
        if selection_topks is None:
            selection_topks = {1, 5}
        self._selection_topks = selection_topks

        self.knowledge = knowledge
        self.reset()

    def reset(self):
        self._detection_tp = 0.0
        self._detection_fp = 0.0
        self._detection_tn = 0.0
        self._detection_fn = 0.0

        self._domain_selection_mrr5 = 0.0
        self._domain_selection_r = {k: 0.0 for k in self._selection_topks}

        self._entity_selection_mrr5 = 0.0
        self._entity_selection_r = {k: 0.0 for k in self._selection_topks}

        self._selection_mrr5 = 0.0
        self._selection_r = {k: 0.0 for k in self._selection_topks}

        self._generation_bleu1 = 0.0
        self._generation_bleu2 = 0.0
        self._generation_bleu3 = 0.0
        self._generation_bleu4 = 0.0
        self._generation_meteor = 0.0
        self._generation_rouge_1 = 0.0
        self._generation_rouge_2 = 0.0
        self._generation_rouge_l = 0.0

        if self.knowledge is not None:
            self._knowledge_bleu1 = 0.0
            self._knowledge_bleu2 = 0.0
            self._knowledge_bleu3 = 0.0
            self._knowledge_bleu4 = 0.0
            self._knowledge_meteor = 0.0
            self._knowledge_rouge_l = 0.0
            self._knowledge_rouge_1 = 0.0
            self._knowledge_rouge_2 = 0.0

    def _match(self, ref_knowledge, pred_knowledge, fields=None):
        if fields is None:
            fields = ['domain', 'entity_id', 'doc_id']
        result = []
        for pred in pred_knowledge:
            matched = True
            for ref in ref_knowledge:
                for field in fields:
                    if field not in pred or field not in ref or pred[field] != ref[field]:
                        matched = False
            result.append(matched)
        return result

    def _reciprocal_rank(self, ref_knowledge, hyp_knowledge, k=5, fields=None):
        relevance = self._match(ref_knowledge, hyp_knowledge, fields)[:k]

        if True in relevance:
            idx = relevance.index(True)
            result = 1.0/(idx+1)
        else:
            result = 0.0

        return result

    def _recall_at_k(self, ref_knowledge, hyp_knowledge, k=5, fields=None):
        relevance = self._match(ref_knowledge, hyp_knowledge, fields)[:k]

        if True in relevance:
            result = 1.0
        else:
            result = 0.0

        return result

    def _normalize_text(self, text):
        result = text.lower()
        result = RE_PUNC.sub(' ', result)
        result = RE_ART.sub(' ', result)
        result = ' '.join(result.split())

        return result

    def _bleu(self, ref_response, hyp_response, n=4):
        ref_tokens = self._normalize_text(ref_response).split()
        hyp_tokens = self._normalize_text(hyp_response).split()

        weights = [1.0/n] * n

        score = sentence_bleu([ref_tokens], hyp_tokens, weights)

        return score

    def _meteor(self, ref_response, hyp_response):
        score = single_meteor_score(
            ref_response, hyp_response, self._normalize_text)

        return score

    def _rouge(self, ref_response, hyp_response, mode='l'):
        ref_response = self._normalize_text(ref_response)
        hyp_response = self._normalize_text(hyp_response)

        if len(hyp_response.strip()) == 0:
            return 0.0

        rouge = Rouge(metrics=["rouge-l", "rouge-n"], max_n=2)

        if mode == 'l':
            score = rouge.get_scores(hyp_response, ref_response)[
                'rouge-l']['f']
        elif mode == 1:
            score = rouge.get_scores(hyp_response, ref_response)[
                'rouge-1']['f']
        elif mode == 2:
            score = rouge.get_scores(hyp_response, ref_response)[
                'rouge-2']['f']
        else:
            raise ValueError("unsupported mode: %s" % mode)

        return score

    def update(self, ref_obj, hyp_obj):
        if ref_obj['target'] is True:
            if hyp_obj['target'] is True:
                self._detection_tp += 1

                if 'knowledge' in hyp_obj and 'domain' in hyp_obj['knowledge'][0]:
                    self._domain_selection_mrr5 += self._reciprocal_rank(
                        ref_obj['knowledge'], hyp_obj['knowledge'], 5, ['domain'])
                    for k in self._domain_selection_r.keys():
                        self._domain_selection_r[k] += self._recall_at_k(
                            ref_obj['knowledge'], hyp_obj['knowledge'], k, ['domain'])

                if 'knowledge' in hyp_obj and 'domain' in hyp_obj['knowledge'][0] and 'entity_id' in hyp_obj['knowledge'][0]:
                    self._entity_selection_mrr5 += self._reciprocal_rank(
                        ref_obj['knowledge'], hyp_obj['knowledge'], 5, ['domain', 'entity_id'])
                    for k in self._entity_selection_r.keys():
                        self._entity_selection_r[k] += self._recall_at_k(
                            ref_obj['knowledge'], hyp_obj['knowledge'], k, ['domain', 'entity_id'])

                if 'knowledge' in hyp_obj and 'domain' in hyp_obj['knowledge'][0] and 'entity_id' in hyp_obj['knowledge'][0] and 'doc_id' in hyp_obj['knowledge'][0]:
                    self._selection_mrr5 += self._reciprocal_rank(
                        ref_obj['knowledge'], hyp_obj['knowledge'], 5, ['domain', 'entity_id', 'doc_id'])
                    for k in self._selection_r.keys():
                        self._selection_r[k] += self._recall_at_k(
                            ref_obj['knowledge'], hyp_obj['knowledge'], k, ['domain', 'entity_id', 'doc_id'])

                if 'response' in hyp_obj:
                    self._generation_bleu1 += self._bleu(
                        ref_obj['response'], hyp_obj['response'], 1)
                    self._generation_bleu2 += self._bleu(
                        ref_obj['response'], hyp_obj['response'], 2)
                    self._generation_bleu3 += self._bleu(
                        ref_obj['response'], hyp_obj['response'], 3)
                    self._generation_bleu4 += self._bleu(
                        ref_obj['response'], hyp_obj['response'], 4)
                    self._generation_meteor += self._meteor(
                        ref_obj['response'], hyp_obj['response'])
                    self._generation_rouge_l += self._rouge(
                        ref_obj['response'], hyp_obj['response'], 'l')
                    self._generation_rouge_1 += self._rouge(
                        ref_obj['response'], hyp_obj['response'], 1)
                    self._generation_rouge_2 += self._rouge(
                        ref_obj['response'], hyp_obj['response'], 2)

                if self.knowledge is not None:
                    knowledge = ref_obj["knowledge"][0]
                    knowledge = self.knowledge.get_doc(
                        knowledge["domain"],
                        knowledge["entity_id"],
                        knowledge["doc_id"]
                    )
                    knowledge = knowledge["doc"]["body"]
                    self._knowledge_bleu1 += self._bleu(
                        knowledge, hyp_obj['response'], 1)
                    self._knowledge_bleu2 += self._bleu(
                        knowledge, hyp_obj['response'], 2)
                    self._knowledge_bleu3 += self._bleu(
                        knowledge, hyp_obj['response'], 3)
                    self._knowledge_bleu4 += self._bleu(
                        knowledge, hyp_obj['response'], 4)
                    self._knowledge_meteor += self._meteor(
                        knowledge, hyp_obj['response'])
                    self._knowledge_rouge_l += self._rouge(
                        knowledge, hyp_obj['response'], 'l')
                    self._knowledge_rouge_1 += self._rouge(
                        knowledge, hyp_obj['response'], 1)
                    self._knowledge_rouge_2 += self._rouge(
                        knowledge, hyp_obj['response'], 2)
            else:
                self._detection_fn += 1
        else:
            if hyp_obj['target'] is True:
                self._detection_fp += 1
            else:
                self._detection_tn += 1

    def _compute(self, score_sum):
        if self._detection_tp + self._detection_fp > 0.0:
            score_p = score_sum/(self._detection_tp + self._detection_fp)
        else:
            score_p = 0.0

        if self._detection_tp + self._detection_fn > 0.0:
            score_r = score_sum/(self._detection_tp + self._detection_fn)
        else:
            score_r = 0.0

        if score_p + score_r > 0.0:
            score_f = 2*score_p*score_r/(score_p+score_r)
        else:
            score_f = 0.0

        return (score_p, score_r, score_f)

    def scores(self):
        detection_p, detection_r, detection_f = self._compute(
            self._detection_tp)

        _, _, domain_selection_mrr5_f = self._compute(
            self._domain_selection_mrr5)
        domain_selection_rk_f = {}
        for k, val in self._domain_selection_r.items():
            domain_selection_rk_f[k] = self._compute(val)[2]

        _, _, entity_selection_mrr5_f = self._compute(
            self._entity_selection_mrr5)
        entity_selection_rk_f = {}
        for k, val in self._entity_selection_r.items():
            entity_selection_rk_f[k] = self._compute(val)[2]

        _, _, selection_mrr5_f = self._compute(self._selection_mrr5)
        selection_rk_f = {}
        for k, val in self._selection_r.items():
            selection_rk_f[k] = self._compute(val)[2]

        _, _, generation_bleu1_f = self._compute(self._generation_bleu1)
        _, _, generation_bleu2_f = self._compute(self._generation_bleu2)
        _, _, generation_bleu3_f = self._compute(self._generation_bleu3)
        _, _, generation_bleu4_f = self._compute(self._generation_bleu4)
        _, _, generation_meteor_f = self._compute(self._generation_meteor)
        _, _, generation_rouge_l_f = self._compute(self._generation_rouge_l)
        _, _, generation_rouge_1_f = self._compute(self._generation_rouge_1)
        _, _, generation_rouge_2_f = self._compute(self._generation_rouge_2)

        if self.knowledge is not None:
            _, _, knowledge_bleu1_f = self._compute(self._knowledge_bleu1)
            _, _, knowledge_bleu2_f = self._compute(self._knowledge_bleu2)
            _, _, knowledge_bleu3_f = self._compute(self._knowledge_bleu3)
            _, _, knowledge_bleu4_f = self._compute(self._knowledge_bleu4)
            _, _, knowledge_meteor_f = self._compute(self._knowledge_meteor)
            _, _, knowledge_rouge_l_f = self._compute(self._knowledge_rouge_l)
            _, _, knowledge_rouge_1_f = self._compute(self._knowledge_rouge_1)
            _, _, knowledge_rouge_2_f = self._compute(self._knowledge_rouge_2)

        scores = {
            'detection': {
                'prec': detection_p,
                'rec': detection_r,
                'f1': detection_f
            },
            'domain_selection': {
                'mrr@5': domain_selection_mrr5_f,
            },
            'entity_selection': {
                'mrr@5': entity_selection_mrr5_f,
            },
            'selection': {
                'mrr@5': selection_mrr5_f,
            },
        }
        if self.knowledge is not None:
            scores['generation'] = {
                'bleu-1': [generation_bleu1_f, knowledge_bleu1_f],
                'bleu-2': [generation_bleu2_f, knowledge_bleu2_f],
                'bleu-3': [generation_bleu3_f, knowledge_bleu3_f],
                'bleu-4': [generation_bleu4_f, knowledge_bleu4_f],
                'meteor': [generation_meteor_f, knowledge_meteor_f],
                'rouge_1': [generation_rouge_1_f, knowledge_rouge_1_f],
                'rouge_2': [generation_rouge_2_f, knowledge_rouge_2_f],
                'rouge_l': [generation_rouge_l_f, knowledge_rouge_l_f],
            }
        else:
            scores['generation'] = {
                'bleu-1': generation_bleu1_f,
                'bleu-2': generation_bleu2_f,
                'bleu-3': generation_bleu3_f,
                'bleu-4': generation_bleu4_f,
                'meteor': generation_meteor_f,
                'rouge_1': generation_rouge_1_f,
                'rouge_2': generation_rouge_2_f,
                'rouge_l': generation_rouge_l_f,
            }

        for k, val in sorted(domain_selection_rk_f.items(), key=lambda x: x[0]):
            scores['domain_selection'][f"r@{k}"] = val
        for k, val in sorted(entity_selection_rk_f.items(), key=lambda x: x[0]):
            scores['entity_selection'][f"r@{k}"] = val
        for k, val in sorted(selection_rk_f.items(), key=lambda x: x[0]):
            scores['selection'][f"r@{k}"] = val

        return scores


def main(argv):
    print(argv)
    parser = argparse.ArgumentParser(
        description='Evaluate the system outputs.')

    parser.add_argument('--split', dest='split', action='store', metavar='DATASET', choices=[
                        'train', 'validation', 'test'], required=True, help='The dataset to analyze')
    parser.add_argument('--dataset', dest='dataset', action='store', metavar='PATH', required=True,
                        help='Will look for corpus in <dataroot>/<dataset>/...')
    parser.add_argument('--outfile', dest='outfile', action='store', metavar='JSON_FILE', required=True,
                        help='File containing output JSON')
    parser.add_argument('--scorefile', dest='scorefile', action='store', metavar='JSON_FILE', required=True,
                        help='File containing scores')
    parser.add_argument('--dataset_filter_json', dest='dataset_filter_json',
                        action='store', metavar='JSON_FILE', default=None)

    args = parser.parse_args()

    with open(args.outfile, 'r') as f:
        output = json.load(f)

    dataset_filter_dict = None
    if args.dataset_filter_json is not None:
        with open(args.dataset_filter_json, "r") as f:
            dataset_filter_dict = json.load(f)

    data = DatasetWalker(args.dataset, args.split, dataset_filter_dict)

    max_knowledge_length = max([len(pred['knowledge'])
                               for pred in output if 'knowledge' in pred], default=0)
    metric = Metric(selection_topks=set(selection_range(max_knowledge_length)))

    for (instance, ref), pred in zip(data, output):
        metric.update(ref, pred)

    scores = metric.scores()

    with open(args.outfile, 'r') as f:
        print(args.outfile)
        output = json.load(f)

    data = load_dataset(args.dataset, "evaluation", split=args.split)

    knowledge = []
    preds = []
    refs = []
    
    print(len(data))
    print(len(output))
    assert len(output) == len(data)
    for pred, ref in zip(output, data):
        if pred["target"] and ref["target"]:
            preds.append(pred["response"])
            refs.append(ref["response"])
            knowledge.append(ref["knowledge"][0]["body"])
    refs = [refs]

    if len(preds) > 0:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        roguel = [scorer.score(ref, pred)['rougeL'].fmeasure for ref, pred in zip(knowledge, preds)]
        scores["generation"]["knowledge_ROGUEL_f"] = np.mean(roguel)
        roguel = [scorer.score(ref, pred)['rougeL'].precision for ref, pred in zip(knowledge, preds)]
        scores["generation"]["knowledge_ROGUEL_p"] = np.mean(roguel)


        f1s = [f1_score(grounding, pred)
               for grounding, pred in zip(knowledge, preds)]
        scores["generation"]["f1"] = np.mean(f1s)
        

        ref_lens = [len(ref[0].split(" ")) for ref in refs]
        pred_lens = [len(pred.split(" ")) for pred in preds]
        scores["generation"]["length_ratio"] = np.mean(
            ref_lens) / np.mean(pred_lens)

        bleu = sacrebleu.corpus_bleu(preds, refs)
        scores["generation"].update({
            "sacrebleu": bleu.score,
        })
        ngram_divs = []
        ngram_repetitions = []

        for N in range(1, 6):
            ngram_freqs, unique_ngrams, ngram_len, repetitions = ngram_stats(
                preds, N)
            ngram_divs.append(unique_ngrams / ngram_len)
            ngram_repetitions.append(repetitions / ngram_len)

        scores["generation"]["ngram_div"] = np.mean(ngram_divs)
        scores["generation"]["repetition"] = np.mean(ngram_repetitions)

    with open(args.scorefile, 'w') as out:
        json.dump(scores, out, indent=2)


if __name__ == "__main__":
    main(sys.argv)
