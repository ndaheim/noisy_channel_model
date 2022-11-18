# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""
DSTC9 Track 1 - Beyond Domain APIs: Task-oriented Conversational Modeling with Unstructured Knowledge Access - Dataset
"""

from __future__ import absolute_import, division, print_function

import json
from typing import List, Optional

import datasets

from .base import DSTCBase
from .ctrl_mixin import CTRLMixin


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@article{kim2020domain,
  title={Beyond Domain APIs: Task-oriented Conversational Modeling with Unstructured Knowledge Access},
  author={Seokhwan Kim and Mihail Eric and Karthik Gopalakrishnan and Behnam Hedayatnia and Yang Liu and Dilek Hakkani-Tur},
  journal={arXiv preprint arXiv:2006.03533}
  year={2020}
}
"""

_DESCRIPTION = """\

"""

_HOMEPAGE = "https://github.com/alexa/alexa-with-dstc9-track1-dataset"


_BASE_URL = "https://raw.githubusercontent.com/alexa/alexa-with-dstc9-track1-dataset/master"
_URLs = {
    'train': {
        'logs': f'{_BASE_URL}/data/train/logs.json',
        'labels': f'{_BASE_URL}/data/train/labels.json',
        'knowledge': f'{_BASE_URL}/data/knowledge.json',
    },
    'val': {
        'logs': f'{_BASE_URL}/data/val/logs.json',
        'labels': f'{_BASE_URL}/data/val/labels.json',
        'knowledge': f'{_BASE_URL}/data/knowledge.json',
    },
    'test': {
        'logs': f'{_BASE_URL}/data_eval/test/logs.json',
        'labels': f'{_BASE_URL}/data_eval/test/labels.json',
        'knowledge': f'{_BASE_URL}/data_eval/knowledge.json',
    }
}

class DSTC9Track1(DSTCBase, datasets.GeneratorBasedBuilder, CTRLMixin):


    def _info(self):

        if self.config.name == "detection":
            features = datasets.Features(
                {
                    "id": datasets.Value('string'),
                    "target": datasets.Value("bool"),
                    "turns": [
                        {
                            "speaker": datasets.Value("string"),
                            "text": datasets.Value("string"),
                        }
                    ],
                    "source": datasets.Value('string'),
                }
            )
        elif self.config.name == "selection":
            features = datasets.Features(
                {
                    "id": datasets.Value('string'),
                    "target": datasets.Value("bool"),
                    "turns": [
                        {
                            "speaker": datasets.Value("string"),
                            "text": datasets.Value("string"),
                        }
                    ],
                    "knowledge": [
                        {
                            "domain": datasets.Value("string"),
                            "entity_id": datasets.Value("string"),
                            "doc_id": datasets.Value("int32"),
                            "score": datasets.Value("float"),
                        }
                    ],
                    "entity_candidates": [{
                        "id": datasets.Value('string'),
                        "domain": datasets.Value("string"),
                        "city": datasets.Value("string"),
                        "entity_id": datasets.Value("string"),
                        "entity_name": datasets.Value("string"),
                    }],
                    "source": datasets.Value('string'),
                }
            )
        elif self.config.name == "selection_search":
            features = datasets.Features(
                {
                    "id": datasets.Value('string'),
                    "target": datasets.Value("bool"),
                    "turns": [
                        {
                            "speaker": datasets.Value("string"),
                            "text": datasets.Value("string"),
                        }
                    ],
                    "entity_candidates": [{
                        "id": datasets.Value('string'),
                        "domain": datasets.Value("string"),
                        "city": datasets.Value("string"),
                        "entity_id": datasets.Value("string"),
                        "entity_name": datasets.Value("string"),
                    }],
                    "source": datasets.Value('string'),
                }
            )
        elif self.config.name == "selection_reranking":
            features = datasets.Features(
                {
                    "id": datasets.Value('string'),
                    "target": datasets.Value("bool"),
                    "turns": [
                        {
                            "speaker": datasets.Value("string"),
                            "text": datasets.Value("string"),
                        }
                    ],
                    "knowledge": [
                        {
                            "domain": datasets.Value("string"),
                            "entity_id": datasets.Value("string"),
                            "doc_id": datasets.Value("int32"),
                            "score": datasets.Value("float"),
                        }
                    ],
                    "knowledge_preds": [
                        {
                            "domain": datasets.Value("string"),
                            "entity_id": datasets.Value("string"),
                            "doc_id": datasets.Value("int32"),
                            "score": datasets.Value("float"),
                        }
                    ],
                    "entity_candidates": [{
                        "id": datasets.Value('string'),
                        "domain": datasets.Value("string"),
                        "city": datasets.Value("string"),
                        "entity_id": datasets.Value("string"),
                        "entity_name": datasets.Value("string"),
                    }],
                    "source": datasets.Value('string'),
                }
            )
        elif self.config.name == "generation_search":
            features = datasets.Features(
                {
                    "id": datasets.Value('string'),
                    "target": datasets.Value("bool"),
                    "turns": [
                        {
                            "speaker": datasets.Value("string"),
                            "text": datasets.Value("string"),
                        }
                    ],
                    "knowledge": [
                        {
                            "domain": datasets.Value("string"),
                            "entity_id": datasets.Value("string"),
                            "doc_id": datasets.Value("int32"),
                            "entity_name": datasets.Value("string"),
                            "title": datasets.Value("string"),
                            "body": datasets.Value("string"),
                            "score": datasets.Value("float"),
                        }
                    ],
                    "entity_candidates": [{
                        "id": datasets.Value('string'),
                        "domain": datasets.Value("string"),
                        "city": datasets.Value("string"),
                        "entity_id": datasets.Value("string"),
                        "entity_name": datasets.Value("string"),
                    }],
                    "source": datasets.Value('string'),
                }
            )
        elif self.config.name in ["evaluation", "generation"]:
            features = datasets.Features(
                {
                    "id": datasets.Value('string'),
                    "target": datasets.Value("bool"),
                    "knowledge": [
                        {
                            "domain": datasets.Value("string"),
                            "entity_id": datasets.Value("string"),
                            "doc_id": datasets.Value("int32"),
                            "entity_name": datasets.Value("string"),
                            "title": datasets.Value("string"),
                            "body": datasets.Value("string"),
                            "score": datasets.Value("float"),
                        }
                    ],
                    "turns": [
                        {
                            "speaker": datasets.Value("string"),
                            "text": datasets.Value("string"),
                        }
                    ],
                    "response": datasets.Value("string"),
                    "source": datasets.Value('string'),
                }
            )
        elif self.config.name == "control":
            features = datasets.Features(
                {
                    "id": datasets.Value('string'),
                    "target": datasets.Value("bool"),
                    "knowledge": [
                        {
                            "domain": datasets.Value("string"),
                            "entity_id": datasets.Value("string"),
                            "doc_id": datasets.Value("int32"),
                            "entity_name": datasets.Value("string"),
                            "title": datasets.Value("string"),
                            "body": datasets.Value("string"),
                            "score": datasets.Value("float"),
                        }
                    ],
                    "turns": [
                        {
                            "speaker": datasets.Value("string"),
                            "text": datasets.Value("string"),
                        }
                    ],
                    "response": datasets.Value("string"),
                    "control_tokens": datasets.Value("string"),
                    "source": datasets.Value('string'),
                }
            )
        elif self.config.name in ["knowledge", "knowledge_entities", "knowledge_domains"]:
            features = datasets.Features(
                {
                    "id": datasets.Value('string'),
                    "domain": datasets.Value("string"),
                    **({
                        "city": datasets.Value("string"),
                        "entity_id": datasets.Value("string"),
                        "entity_name": datasets.Value("string"),
                    } if self.config.name in ["knowledge", "knowledge_entities"] else {}),
                    **({
                        "doc_id": datasets.Value("int32"),
                        "title": datasets.Value("string"),
                        "body": datasets.Value("string"),
                    } if self.config.name == "knowledge" else {}),
                }
            )
        else:
            assert False, f"Unexpected config name: {self.config.name}"

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )


    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        urls_to_download = _URLs
        downloaded_files = self._download_files(urls_to_download, self.config.data_files, dl_manager)

        return [
            datasets.SplitGenerator(name=ds_split, gen_kwargs=downloaded_files[split])
            for ds_split, split in (
                (datasets.Split.TRAIN, 'train'),
                (datasets.Split.VALIDATION, 'val'),
                (datasets.Split.TEST, 'test')
            )
        ]
        
    def _filter(self, label):
        if label is None:
            return True
        if self.config.dataset_filter_dict is not None:
            for key, value in self.config.dataset_filter_dict.items():
                if key in label and label[key] != value:
                    return False
        return True

    def _generate_examples(self, logs, knowledge, labels=None, predictions=None):
        with open(logs) as fp:
            logs_data = json.load(fp)
        for log in logs_data:
            for turn in log:
                if "nbest" in turn:
                    del turn["nbest"]
        if labels is not None:
            with open(labels) as fp:
                labels_data = json.load(fp)
        else:
            labels_data = [None] * len(logs_data)
        with open(knowledge) as fp:
            knowledge_data = json.load(fp)
        if predictions is None:
            predictions_data = [None] * len(labels_data)
        else:
            with open(predictions) as fp:
                predictions_data = json.load(fp)

        if self.config.name in ["knowledge", "knowledge_entities", "knowledge_domains"]:
            yield from self._generate_knowledge(knowledge_data)
        else:
            if len(predictions_data) != len(labels_data):
                preds_iter = iter(predictions_data)
                new_predictions_data = []
                for label in labels_data:
                    if not self._filter(label):
                        new_predictions_data.append({
                            'target': False,
                        })
                    else:
                        new_predictions_data.append(next(preds_iter))
                predictions_data = new_predictions_data
                assert len(predictions_data) == len(labels_data)
                
            data = []

            for i, log, label, pred in zip(range(len(logs_data)), logs_data, labels_data, predictions_data):

                if not self._filter(label):
                    continue

                if pred is not None:
                    label = pred

                if label is None:
                    label = {
                        'target': False,
                    }

                if label["target"] and self.config.name not in ['detection', 'selection_search']:
                    label["knowledge"][0]["entity_id"] = str(label["knowledge"][0]["entity_id"])

                if 'source' not in label:
                    if self.__class__.__name__ == 'DSTC10Track2':
                        source = 'sf_spoken_asr'
                    else:
                        source = 'multiwoz'
                else:
                    source = label['source']

                x = {
                    'id': str(i),
                    'target': label['target'],
                    'turns': log,
                    'source': source,
                }
                if not x["target"] and self.config.name not in ['detection', 'evaluation', 'selection_reranking']:
                    continue

                if x["target"]:
                    if self.config.name in ['evaluation', 'selection', 'selection_reranking', 'generation', 'generation_search', "control"]:
                        x['knowledge'] = label['knowledge']
                        if self.config.name not in ["generation", "evaluation", "control"]:
                            x['entity_candidates'] = label.get('entity_candidates', [])
                        if self.config.name in ['selection_reranking']:
                            x['knowledge_preds'] = label.get('knowledge_preds', [])
                    if self.config.name == 'selection_search':
                        x['entity_candidates'] = label.get('entity_candidates', [])
                    if self.config.name in ['evaluation', 'generation', 'generation_search', "control"]:

                        for k in range(len(x['knowledge'])):
                            domain, entity_id, doc_id = x['knowledge'][k].get('domain'), x['knowledge'][k].get('entity_id'), x['knowledge'][k].get('doc_id')
                            if domain is not None and entity_id is not None and doc_id is not None:
                                x['knowledge'][k].update(knowledge_data[domain][str(entity_id)]['docs'][str(doc_id)])
                            else:
                                x['knowledge'][k]['title'] = ''
                                x['knowledge'][k]['body'] = ''
                            if domain is not None and entity_id is not None:
                                x['knowledge'][k]['entity_name'] = knowledge_data[domain][str(entity_id)]['name']
                            else:
                                x['knowledge'][k]['entity_name'] = ''
                            if doc_id is None:
                                x['knowledge'][k]['doc_id'] = -1
                            if entity_id is None:
                                x['knowledge'][k]['entity_id'] = ''
                            if domain is None:
                                x['knowledge'][k]['domain'] = ''
                            if 'score' not in x['knowledge'][k]:
                                x['knowledge'][k]['score'] = 1.0
                    elif self.config.name in ['selection']:
                        for k in range(len(x['knowledge'])):
                            if 'score' not in x['knowledge'][k]:
                                x['knowledge'][k]['score'] = 1.0
                    elif self.config.name in ['selection_reranking']:
                        for k in range(len(x['knowledge_preds'])):
                            if 'score' not in x['knowledge_preds'][k]:
                                x['knowledge_preds'][k]['score'] = 1.0
                        for k in range(len(x['knowledge'])):
                            if 'score' not in x['knowledge'][k]:
                                x['knowledge'][k]['score'] = 1.0


                    if self.config.name in ['evaluation', 'generation', "control"]:
                        x['response'] = label['response'] if 'response' in label else ''

                elif self.config.name == 'evaluation':
                    x["knowledge"] = []
                    x["response"] = ""
                elif self.config.name == 'selection_reranking':
                    x["knowledge"] = label.get('knowledge', [])
                    x["knowledge_preds"] = label.get('knowledge_preds', [])
                    x['entity_candidates'] = label.get('entity_candidates', [])
                    
                data.append(x)
            
            if self.config.name == "control":
                data = self._add_control_tokens(data)
            
            for sample in data:
                yield sample["id"], sample

    def _generate_knowledge(self, knowledge_data):
        for domain, domain_knowledge in knowledge_data.items():
            if self.config.name in ["knowledge", "knowledge_entities"]:
                for entity_id, entity_knowledge in domain_knowledge.items():
                    entity_name = entity_knowledge['name']
                    city = str(entity_knowledge.get('city', 'Cambridge'))
                    if self.config.dataset_filter_dict is not None and self.config.dataset_filter_dict.get('city', city) != city:
                        continue
                    if self.config.name == "knowledge":
                        for doc_id, doc_knowledge in entity_knowledge['docs'].items():
                            id_ = f"{domain}__{entity_id}__{doc_id}"
                            yield id_, {
                                'id': id_,
                                'domain': domain,
                                'city': city,
                                'entity_id': entity_id,
                                'doc_id': doc_id,
                                'entity_name': entity_name,
                                'title': doc_knowledge['title'],
                                'body': doc_knowledge['body'],
                            }
                    else:
                        id_ = f"{domain}__{entity_id}"
                        yield id_, {
                            'id': id_,
                            'domain': domain,
                            'city': city,
                            'entity_id': entity_id,
                            'entity_name': entity_name,
                        }
            else:
                id_ = f"{domain}"
                yield id_, {
                    'id': id_,
                    'domain': domain,
                }

    def _download_files(self, urls, data_files, dl_manager):
        if data_files is not None:
            if isinstance(urls, str):
                urls = {
                    "data": urls,
                    "train": {},
                    "val": {},
                    "test": {}
                }
            for split, update_dict in data_files.items():
                if isinstance(update_dict, dict):
                    for key, value in update_dict.items():
                        urls[split][key] = value
                elif isinstance(update_dict, datasets.data_files.DataFilesList):
                    file_path = update_dict[0]
                    urls[split]['predictions'] = file_path
                else:
                    assert type(str(update_dict)) == str
                    urls[split]['predictions'] = update_dict

        return dl_manager.download_and_extract(urls)
