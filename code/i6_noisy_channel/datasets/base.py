from __future__ import absolute_import, division, print_function

import json
import os
from typing import List, Optional

import datasets

from datasets import BuilderConfig
from datasets.features import Features
from datasets.fingerprint import Hasher


MAX_DIRECTORY_NAME_LENGTH = 255


class DSTCBuilderConfig(BuilderConfig):
    dataset_filter_dict: dict = None    
    dataset_transformations: List[str] = None


    def create_config_id(self, config_kwargs: dict, custom_features: Optional[Features] = None, use_auth_token=False) -> str:
        """
        The config id is used to build the cache directory.
        By default it is equal to the config name.
        However the name of a config is not sufficent to have a unique identifier for the dataset being generated since
        it doesn't take into account:
        - the config kwargs that can be used to overwrite attributes
        - the custom features used to write the dataset
        - the data_files for json/text/csv/pandas datasets
        Therefore the config id is just the config name with an optional suffix based on these.
        """
        # Possibly add a suffix to the name to handle custom features/data_files/config_kwargs
        suffix: Optional[str] = None
        config_kwargs_to_add_to_suffix = dict(config_kwargs)
        # name and version are already used to build the cache directory
        config_kwargs_to_add_to_suffix.pop("name", None)
        config_kwargs_to_add_to_suffix.pop("version", None)
        # data files are handled differently
        config_kwargs_to_add_to_suffix.pop("data_files", None)
        # data dir is ignored (when specified it points to the manually downloaded data)
        config_kwargs_to_add_to_suffix.pop("data_dir", None)
        if config_kwargs_to_add_to_suffix:
            # we don't care about the order of the kwargs
            config_kwargs_to_add_to_suffix = {
                k: config_kwargs_to_add_to_suffix[k] for k in sorted(config_kwargs_to_add_to_suffix)
            }
            if all(isinstance(v, (str, bool, int, float)) for v in config_kwargs_to_add_to_suffix.values()):
                suffix = ",".join(
                    str(k) + "=" + urllib.parse.quote_plus(str(v)) for k, v in config_kwargs_to_add_to_suffix.items()
                )
                if len(suffix) > 32:  # hash if too long
                    suffix = Hasher.hash(config_kwargs_to_add_to_suffix)
            else:
                suffix = Hasher.hash(config_kwargs_to_add_to_suffix)

        if self.data_files is not None:
            m = Hasher()
            if suffix:
                m.update(suffix)
            if isinstance(self.data_files, str):
                data_files = {"train": [self.data_files]}
            elif isinstance(self.data_files, (tuple, list)):
                data_files = {"train": self.data_files}
            elif isinstance(self.data_files, dict):
                data_files = {
                    str(key): files if isinstance(files, (tuple, list)) else [files]
                    for key, files in self.data_files.items()
                }
            else:
                raise ValueError("Please provide a valid `data_files` in `DatasetBuilder`")
            for key in sorted(data_files.keys()):
                m.update(key)
                for data_file in data_files[key]:
                    if isinstance(data_file, dict):
                        for _, value in data_file.items():
                            m.update(os.path.abspath(value))
                            m.update(str(os.path.getmtime(value)))
                    else:
                        m.update(os.path.abspath(data_file))
                        m.update(str(os.path.getmtime(data_file)))
            suffix = m.hexdigest()

        if custom_features is not None:
            m = Hasher()
            if suffix:
                m.update(suffix)
            m.update(custom_features)
            suffix = m.hexdigest()

        if suffix:
            config_id = self.name + "-" + suffix
            if len(config_id) > MAX_DIRECTORY_NAME_LENGTH:
                config_id = self.name + "-" + Hasher.hash(suffix)
            return config_id
        else:
            return self.name

_CITATION = """\
@article{kim2020domain,
  title={Beyond Domain APIs: Task-oriented Conversational Modeling with Unstructured Knowledge Access},
  author={Seokhwan Kim and Mihail Eric and Karthik Gopalakrishnan and Behnam Hedayatnia and Yang Liu and Dilek Hakkani-Tur},
  journal={arXiv preprint arXiv:2006.03533}
  year={2020}
}
"""

_HOMEPAGE = "https://github.com/alexa/alexa-with-dstc10-track2-dataset"


_DESCRIPTION = """\

"""

class DSTCBase(object):
    
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        DSTCBuilderConfig(
            name="detection",
            version=VERSION,
            description="",
        ),
        DSTCBuilderConfig(
            name="control",
            version=VERSION,
            description="",
        ),
        DSTCBuilderConfig(
            name="evaluation",
            version=VERSION,
            description="",
        ),
        DSTCBuilderConfig(
            name="selection",
            version=VERSION,
            description="",
        ),
        DSTCBuilderConfig(
            name="selection_reranking",
            version=VERSION,
            description="",
        ),
        DSTCBuilderConfig(
            name="selection_search",
            version=VERSION,
            description="",
        ),
        DSTCBuilderConfig(
            name="generation",
            version=VERSION,
            description="",
        ),
        DSTCBuilderConfig(
            name="generation_search",
            version=VERSION,
            description="",
        ),
        DSTCBuilderConfig(
            name="knowledge",
            version=VERSION,
            description="",
        ),
        DSTCBuilderConfig(
            name="knowledge_entities",
            version=VERSION,
            description="",
        ),
        DSTCBuilderConfig(
            name="knowledge_domains",
            version=VERSION,
            description="",
        ),
    ]

    DEFAULT_CONFIG_NAME = "detection"


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
                            "score": datasets.Value("float")
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
                    "response": datasets.Value("string"),
                    "source": datasets.Value('string'),
                }
            )
        elif self.config.name == "control":
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
                    # x['entity_candidates'] = []
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