import itertools
import math
import random
import re
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional
from datasets import load_metric, load_dataset, arrow_dataset
from torch.utils.data import Dataset
from transformers import PretrainedConfig, EvalPrediction, AutoModelForSequenceClassification
from transformers.trainer_utils import PredictionOutput

from i6_noisy_channel.arguments import ModelArguments
from i6_noisy_channel.knowledge_utils import build_knowledge_document_register_with_city
from i6_noisy_channel.methods.base import Method
from i6_noisy_channel.preprocessing import create_concatenated_dialog_knowledge_input, \
    process_input, process_knowledge
from i6_noisy_channel.utils import iterate_values_in_nested_dict


class CrossEncoderDataset(Dataset):
    def __init__(self, model_config, args: ModelArguments, tokenizer, query_dataset, documents_dataset, document_register, train_mode=True):
        self.model_config = model_config
        self.args = args
        self.tokenizer = tokenizer
        self.query_dataset = query_dataset
        self.documents_dataset = documents_dataset
        self.train_mode = train_mode

        # Map: [city, domain, entity_id, doc_id] -> idx in documents dataset
        self.document_register = document_register
        self._build_data_infos()

        # Plausibility check
        assert args.selection_level in [
            'all', 'document', 'entity', 'domain', 'domain_entity']
        assert args.selection_level not in [
            'entity', 'domain', 'domain_entity'] or args.num_doc_negatives == 0
        assert args.selection_level not in [
            'domain'] or args.num_entity_negatives == 0

    def _get_city_from_source(self, source):
        if source in ['multiwoz']:
            return 'Cambridge'
        if source in ['sf_written', 'sf_spoken', 'sf_spoken_asr']:
            return 'San Francisco'
        if source in ['simulated']:
            return '*'
        assert False, f'Unknown source: {source}'

    def _build_data_infos(self):
        data_infos = []
        query_slices = []
        query_slice_counter = 0
        data_scores = []
        for query_idx, query_item in enumerate(self.query_dataset):
            city = self._get_city_from_source(query_item['source'])
            if self.args.selection_level in ["all", "domain", "domain_entity"]:
                data_info = list(iterate_values_in_nested_dict(
                    self.document_register[city]))
            elif self.args.selection_level == 'entity':
                if isinstance(query_item['domain'], str):
                    data_info = list(
                        self.document_register[city][query_item['domain']].values())
                else:
                    data_info_with_scores = [(doc, score) for domain, score in zip(
                        query_item['domain'], query_item['score']) for doc in self.document_register[city][domain].values()]
                    data_info, scores = zip(*data_info_with_scores)
                    data_scores.extend(scores)
            elif self.args.selection_level == 'document':
                if isinstance(query_item['domain'], str):
                    data_info = list(
                        self.document_register[city][query_item['domain']][query_item['entity_id']].values())
                else:
                    data_info_with_scores = [
                        (doc, score) for domain, entity, score in zip(query_item['domain'], query_item['entity_id'], query_item['score'])
                        for doc in self.document_register[city][domain][entity].values()
                    ]
                    data_info, scores = zip(*data_info_with_scores)
                    data_scores.extend(scores)
            else:
                assert False
            data_infos.extend(
                list(zip(itertools.cycle([query_idx]), data_info)))
            query_slices.append(
                (query_slice_counter, query_slice_counter + len(data_info)))
            query_slice_counter += len(data_info)
        self.data_infos = data_infos
        self.query_slices = query_slices
        self.data_scores = data_scores

    def _get_number_of_documents_per_sample(self):
        return self.args.num_domain_negatives + self.args.num_entity_negatives + self.args.num_doc_negatives + 1

    def _get_document_index(self, city, domain, entity_id, doc_id):
        entities = self.document_register[city][domain]
        if self.args.selection_level == 'domain':
            return entities
        docs = entities[entity_id]
        if self.args.selection_level in ['entity', 'domain_entity']:
            return docs
        return docs[doc_id]

    def _sample_negative(self, query_item, document_type):
        # Set the sampling level
        city = self._get_city_from_source(query_item['source'])

        negative_sample_level = None
        if document_type < self.args.num_domain_negatives + 1:
            # Domain negatives
            negative_sample_level = "domain"
        elif document_type < self.args.num_entity_negatives + self.args.num_domain_negatives + 1:
            # Entity negatives
            if len(self.document_register[city][query_item['domain']]) > 1:
                negative_sample_level = "entity"
            else:
                negative_sample_level = "domain"
        elif document_type < self.args.num_doc_negatives + self.args.num_entity_negatives + self.args.num_domain_negatives + 1:
            # Doc negatives
            negative_sample_level = "document"

        # Randomly select negatives
        if negative_sample_level == "domain":
            if len(query_item['entity_candidates']) == 0:
                possible_domains = self.document_register[city].keys()
            else:
                # TODO can we remove the hardcoding?
                possible_domains = {'train', 'taxi'} | {
                    c['domain'] for c in query_item['entity_candidates']}
            negative_domain = random.choice(
                list(possible_domains - {query_item['domain']}))
        else:
            negative_domain = query_item['domain']

        if self.args.selection_level in ['domain']:
            negative_entity = None
        elif negative_sample_level in ['entity', 'domain']:
            if len(query_item['entity_candidates']) == 0 or negative_domain in {'train', 'taxi'}:
                possible_entities = self.document_register[city][negative_domain].keys(
                )
            else:
                # TODO can we remove the hardcoding?
                possible_entities = {
                    c['entity_id'] for c in query_item['entity_candidates'] if c['domain'] == negative_domain}
            if negative_sample_level == 'entity':
                negative_entity = random.choice(
                    list(possible_entities - {query_item['entity_id']}))
            elif negative_sample_level == 'domain':
                negative_entity = random.choice(list(possible_entities))
            else:
                assert False
        else:
            negative_entity = query_item['entity_id']

        if self.args.selection_level in ['domain_entity', 'entity']:
            negative_doc = None
        elif negative_sample_level == 'document':
            possible_documents = list(
                self.document_register[city][negative_domain][negative_entity].keys() - {query_item['doc_id']})
            if len(possible_documents) == 0:
                negative_entity = random.choice(list(
                    self.document_register[city][negative_domain].keys() - {query_item['entity_id']}))
                possible_documents = list(
                    self.document_register[city][negative_domain][negative_entity].keys())
            negative_doc = random.choice(possible_documents)
        else:
            negative_doc = random.choice(
                list(self.document_register[city][negative_domain][negative_entity].keys()))

        return self._get_document_index(city, negative_domain, negative_entity, negative_doc)

    def __getitem__(self, index):
        document_index: int
        label: int
        if not self.train_mode:
            query_index, document_index = self.data_infos[index]
            query_item = self.query_dataset[query_index]

            city = self._get_city_from_source(query_item['source'])
            try:
                label_index = self._get_document_index(
                    city, query_item['domain'], query_item['entity_id'], query_item['doc_id'])
                label = int(document_index == label_index)
            except Exception:
                label = 0
        else:
            docs_per_sample = self._get_number_of_documents_per_sample()
            query_index = index // docs_per_sample
            document_type = index % docs_per_sample

            query_item = self.query_dataset[query_index]

            if document_type == 0:
                # Positive
                city = self._get_city_from_source(query_item['source'])
                document_index = self._get_document_index(
                    city, query_item['domain'], query_item['entity_id'], query_item['doc_id'])
                label = 1
            else:
                document_index = self._sample_negative(
                    query_item, document_type)
                label = 0

        input_ids = create_concatenated_dialog_knowledge_input(
            self.args, self.tokenizer, query_item['input_ids'], self.documents_dataset[document_index]['input_ids'])

        return {
            'input_ids': input_ids,
            'labels': label if self.model_config.num_labels == 2 else float(label)
        }

    def __len__(self):
        if not self.train_mode:
            return len(self.data_infos)
        return len(self.query_dataset) * self._get_number_of_documents_per_sample()


class CrossEncoderMethod(Method):
    name = "cross_encoder"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.num_labels not in [1, 2]:
            self.config.num_labels = 2

        self.metrics = [
            load_metric(metric) for metric in ['accuracy']
        ]

    def preprocess_features(self, features):
        input_ids = [
            process_input(self.model_args, turns, self.tokenizer) for turns in features['turns']
        ]

        data = {
            'input_ids': input_ids,
            'id': features['id'],
            'target': features['target'],
            'source': features['source'],
        }
        if 'knowledge' in features:
            if self.model_args.hierarchical_selection_beam_search and 'score' in features['knowledge'][0][0]:
                data.update({
                    'domain': [
                        [
                            x[i]['domain'] for i in range(len(x))
                            if x[i]['score'] > x[0]['score'] * self.model_args.hierarchical_selection_beam_threshold
                        ]
                        for x in features['knowledge']
                    ],
                    'entity_id': [
                        [
                            x[i]['entity_id'] for i in range(len(x))
                            if x[i]['score'] > x[0]['score'] * self.model_args.hierarchical_selection_beam_threshold
                        ]
                        for x in features['knowledge']
                    ],
                    'doc_id': [
                        [
                            x[i]['doc_id'] for i in range(len(x))
                            if x[i]['score'] > x[0]['score'] * self.model_args.hierarchical_selection_beam_threshold
                        ]
                        for x in features['knowledge']
                    ],
                    'score': [
                        [
                            x[i]['score'] for i in range(len(x))
                            if x[i]['score'] > x[0]['score'] * self.model_args.hierarchical_selection_beam_threshold
                        ]
                        for x in features['knowledge']
                    ],
                })
            else:
                data.update({
                    'domain': [x[0]['domain'] for x in features['knowledge']],
                    'entity_id': [x[0]['entity_id'] for x in features['knowledge']],
                    'doc_id': [x[0]['doc_id'] for x in features['knowledge']],
                })
        if 'entity_candidates' in features:
            data.update({
                'entity_candidates': [[{
                    'domain': c['domain'],
                    'entity_id': c['entity_id'],
                } for c in x] for x in features['entity_candidates']]
            })

        if self.model_args.process_all_in_nbest_last_turn:
            if 'nbest' in features['turns'][0][-1]:
                data_nbest = {
                    k: [] for k in data.keys()
                }
                data_nbest['asr_score'] = []
                for i, turns in enumerate(features['turns']):
                    for hyp in turns[-1]['nbest']:
                        current_turns = deepcopy(turns)
                        current_turns[-1]['text'] = hyp['hyp']

                        input_ids = process_input(
                            self.model_args, current_turns, self.tokenizer)
                        data_nbest['input_ids'].append(input_ids)
                        data_nbest['asr_score'].append(hyp['score'])
                        for data_key in data.keys():
                            if data_key == 'input_ids':
                                continue
                            data_nbest[data_key].append(data[data_key][i])
                data = data_nbest

        return data

    def preprocess_documents(self, features):
        if self.data_args.dataset_lowercase_entities:
            features = features.copy()
            if 'entity_name' in features and features['entity_name'] is not None:
                features['entity_name'] = features['entity_name'].lower()
        out = {
            'input_ids': process_knowledge(self.model_args, self.tokenizer, features),
            'domain': features['domain'],
        }
        if 'entity_id' in features:
            out['entity_id'] = features['entity_id']
        if 'doc_id' in features:
            out['doc_id'] = features['doc_id']
        return out

    def get_model_class(self, config: PretrainedConfig):
        return AutoModelForSequenceClassification

    def compute_metrics(self, p: EvalPrediction):
        if self.config.num_labels == 2:
            prediction_ids = np.argmax(p.predictions, axis=-1)
        elif self.config.num_labels == 1:
            prediction_ids = torch.sigmoid(
                torch.tensor(p.predictions))[:, 0] >= 0.5
        results = {}
        for metric in self.metrics:
            results.update(
                metric.compute(predictions=prediction_ids,
                               references=p.label_ids)
            )
        return results

    def postprocess_predictions(self, p: PredictionOutput, dataset):
        if self.config.num_labels == 2:
            scores = torch.log_softmax(torch.tensor(p.predictions), -1)[:, 1]
        elif self.config.num_labels == 1:
            scores = torch.nn.functional.logsigmoid(
                torch.tensor(p.predictions))[:, 0]
        else:
            assert False

        if self.model_args.hierarchical_selection_beam_search and len(dataset.data_scores) > 0:
            assert len(dataset.data_scores) == scores.shape[0]
            scores += np.log(np.array(dataset.data_scores)) * \
                self.model_args.hierarchical_selection_beam_factor

        def get_label(start, scores, indices):
            return [
                {
                    'score': score.exp().item(),
                    **{
                        k: v for k, v in dataset.documents_dataset[dataset.data_infos[index.item() + start][1]].items()
                        if k in ['domain', 'entity_id', 'doc_id']
                    },
                }
                for score, index in zip(scores, indices)
            ]

        if self.model_args.process_all_in_nbest_last_turn and 'asr_score' in dataset.query_dataset.column_names:
            list_of_scores = []
            list_of_asr_scores = []
            list_of_start_pos = []
            prev_id = None
            for query_idx in range(len(dataset.query_dataset)):
                start, end = dataset.query_slices[query_idx]

                row_id = dataset.query_dataset['id'][query_idx]
                if row_id != prev_id:
                    prev_id = row_id
                    list_of_scores.append([])
                    list_of_asr_scores.append([])
                    list_of_start_pos.append(start)
                list_of_scores[-1].append(scores[start:end])
                list_of_asr_scores[-1].append(
                    dataset.query_dataset['asr_score'][query_idx])
            # Combine nbest score
            if self.model_args.selection_nbest_combination_stratey == 'average':
                doc_scores = [torch.logsumexp(torch.stack(
                    score_list), dim=0) - math.log(len(score_list)) for score_list in list_of_scores]
            elif self.model_args.selection_nbest_combination_stratey == 'weighted_by_asr':
                doc_scores = []
                for score_list, asr_scores in zip(list_of_scores, list_of_asr_scores):
                    norm_asr_scores = torch.log_softmax(
                        torch.tensor(asr_scores), 0)
                    doc_scores.append(
                        torch.logsumexp(
                            norm_asr_scores.view(-1, 1) + torch.stack(score_list), dim=0)
                    )
            else:
                assert False, f'Unsupported nbest detection combination strategy: {self.model_args.selection_nbest_combination_stratey}'
            # Get outputs
            return [
                get_label(start, *doc_score.topk(min(len(doc_score),
                          self.model_args.selection_prediction_topk)))
                for start, doc_score in zip(list_of_start_pos, doc_scores)
            ]

        def get_documents(query_idx):
            start, end = dataset.query_slices[query_idx]
            k = min(end - start, self.model_args.selection_prediction_topk)
            topk_values, topk_indices = scores[start:end].topk(k)
            return get_label(start, topk_values, topk_indices)

        return list(map(get_documents, range(len(dataset.query_dataset))))

    def _get_dataset(self, split, config_name=None):
        print(split)
        query_dataset = super()._get_dataset(split, config_name=config_name)
        print(len(query_dataset))
        # raise Exception()
        if config_name in ["evaluation", "detection"]:
            return query_dataset

        # Remove the slice when loading the document dataset
        if self.model_args.selection_level == 'domain':
            document_dataset_config_name = "knowledge_domains"
        elif self.model_args.selection_level in ['entity', 'domain_entity']:
            document_dataset_config_name = "knowledge_entities"
        else:
            document_dataset_config_name = "knowledge"
        document_dataset_split = re.sub(r'^(\w+)(\[.*\])?', r'\1', split)
        document_dataset: arrow_dataset.Dataset = load_dataset(
            self.data_args.dataset_name,
            document_dataset_config_name,
            split=document_dataset_split,
            cache_dir=self.model_args.cache_dir,
            data_files=self.data_args.dataset_data_files,
            dataset_filter_dict=self.data_args.dataset_filter_dict,
        )
        document_register = build_knowledge_document_register_with_city(
            document_dataset)
        old_eval_column_names = document_dataset.column_names
        document_dataset = document_dataset.map(
            self.preprocess_documents,
            batched=False,
            remove_columns=old_eval_column_names,
        )

        return CrossEncoderDataset(
            self.config,
            self.model_args,
            self.tokenizer,
            query_dataset,
            document_dataset,
            document_register,
            self.data_args.is_training,
        )
