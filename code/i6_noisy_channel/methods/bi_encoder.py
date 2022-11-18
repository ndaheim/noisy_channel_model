import copy
import itertools
import random
from dataclasses import dataclass
from typing import Union, Optional, List, Dict

import torch
from datasets import load_metric
from torch.utils.data import Dataset
from transformers import PretrainedConfig, EvalPrediction, PreTrainedTokenizerBase, AutoModel
from transformers.file_utils import PaddingStrategy
from transformers.trainer_utils import PredictionOutput

from i6_noisy_channel.arguments import ModelArguments
from i6_noisy_channel.knowledge_utils import build_knowledge_document_register
from i6_noisy_channel.methods.base import MethodWithDocumentDataset
from i6_noisy_channel.models.bi_encoder import BiEncoderModel, RobertaForBiEncoder
from i6_noisy_channel.preprocessing import process_input, process_knowledge, wrap_with_special_tokens
from i6_noisy_channel.utils import randrange_excluding


class BiEncoderDataset(Dataset):
    def __init__(self, args: ModelArguments, tokenizer, query_dataset, documents_dataset, document_register,
                 train_mode=True):
        self.args = args
        self.tokenizer = tokenizer
        self.query_dataset = query_dataset
        self.documents_dataset = documents_dataset
        self.train_mode = train_mode

        # Map: [domain, entity_id, doc_id] -> idx in documents dataset
        self.document_register = document_register

        # Plausibility check
        assert args.selection_level in [
            'all', 'document', 'entity', 'domain', 'domain_entity']
        assert args.selection_level not in [
            'entity', 'domain', 'domain_entity'] or args.num_doc_negatives == 0
        assert args.selection_level not in [
            'domain'] or args.num_entity_negatives == 0

    def _build_data_infos(self):
        data_infos = []
        query_slices = []
        query_slice_counter = 0

        for query_idx, query_item in enumerate(self.query_dataset):
            if self.args.selection_level in ["all", "domain", "domain_entity"]:
                data_info = list(range(len(self.documents_dataset)))
            elif self.args.selection_level == 'entity':
                data_info = list(
                    self.document_register[query_item['domain']].values())
            elif self.args.selection_level == 'document':
                data_info = list(
                    self.document_register[query_item['domain']][query_item['entity_id']].values())
            else:
                assert False
            data_infos.extend(
                list(zip(itertools.cycle([query_idx]), data_info)))
            query_slices.append(
                (query_slice_counter, query_slice_counter + len(data_info)))
            query_slice_counter += len(data_info)
        self.data_infos = data_infos
        self.query_slices = query_slices

    def _get_document_index(self, domain, entity_id, doc_id):
        entities = self.document_register[domain]
        if self.args.selection_level == 'domain':
            return entities
        docs = entities[entity_id]
        if self.args.selection_level in ['entity', 'domain_entity']:
            return docs
        return docs[doc_id]

    def _sample_negative(self, query_item, document_type):
        if self.args.sample_document_uniform:
            positive_document_index = self._get_document_index(query_item['domain'], query_item['entity_id'],
                                                               query_item['doc_id'])
            sampled_document_index = randrange_excluding(
                0, len(self.documents_dataset), positive_document_index)
            return sampled_document_index

        # Set the sampling level
        negative_sample_level = None
        if document_type < self.args.num_domain_negatives + 1:
            # Domain negatives
            negative_sample_level = "domain"
        elif document_type < self.args.num_entity_negatives + self.args.num_domain_negatives + 1:
            # Entity negatives
            if len(self.document_register[query_item['domain']]) > 1:
                negative_sample_level = "entity"
            else:
                negative_sample_level = "domain"
        elif document_type < self.args.num_doc_negatives + self.args.num_entity_negatives + self.args.num_domain_negatives + 1:
            # Doc negatives
            negative_sample_level = "document"

        # Randomly select negatives
        if negative_sample_level == "domain":
            negative_domain = random.choice(
                list(self.document_register.keys() - {query_item['domain']}))
        else:
            negative_domain = query_item['domain']

        if self.args.selection_level in ['domain']:
            negative_entity = None
        elif negative_sample_level == 'entity':
            negative_entity = random.choice(
                list(self.document_register[negative_domain].keys() - {query_item['entity_id']}))
        elif negative_sample_level == 'domain':
            negative_entity = random.choice(
                list(self.document_register[negative_domain].keys()))
        else:
            negative_entity = query_item['entity_id']

        if self.args.selection_level in ['domain_entity', 'entity']:
            negative_doc = None
        elif negative_sample_level == 'document':
            choices = list(self.document_register[negative_domain][negative_entity].keys() - {query_item['doc_id']})
            if len(choices) == 0:
                # fallback for domains with only one document (e.g. in WoW)
                negative_domain = random.choice(list(self.document_register.keys() - {query_item['domain']}))
                negative_entity = random.choice(list(self.document_register[negative_domain].keys() - {query_item['entity_id']}))
                choices = random.choice(list(self.document_register[negative_domain][negative_entity].keys()))
                
            choices = choices if isinstance(choices, list) else [choices]
            negative_doc = random.choice(choices)
        else:
            negative_doc = random.choice(
                list(self.document_register[negative_domain][negative_entity].keys()))

        return self._get_document_index(negative_domain, negative_entity, negative_doc)

    def __getitem__(self, index):
        query_index = index

        query_item = self.query_dataset[query_index]
        query_input_ids = wrap_with_special_tokens(
            self.tokenizer, query_item['input_ids'])

        positive_document_index = self._get_document_index(query_item['domain'], query_item['entity_id'],
                                                           query_item['doc_id'])

        if self.train_mode:
            if not self.args.sample_dialog_contexts or torch.rand(()).item() <= len(self.documents_dataset) / (len(self.query_dataset) + len(self.documents_dataset)):
                document_type = random.randrange(0,
                                                 self.args.num_domain_negatives + self.args.num_entity_negatives + self.args.num_doc_negatives) + 1

                negative_document_index = self._sample_negative(
                    query_item, document_type)

                positive_document_input_ids = wrap_with_special_tokens(self.tokenizer,
                                                                       self.documents_dataset[positive_document_index][
                                                                           'input_ids'])
                negative_document_input_ids = wrap_with_special_tokens(self.tokenizer,
                                                                       self.documents_dataset[negative_document_index][
                                                                           'input_ids'])
            else:
                # Choose document as anchor and sample negative dialog
                positive_document_input_ids = query_input_ids
                query_input_ids = wrap_with_special_tokens(self.tokenizer,
                                                           self.documents_dataset[positive_document_index][
                                                               'input_ids'])
                # Negative dialog
                negative_query_index = randrange_excluding(
                    0, len(self.query_dataset), query_index)
                negative_document_input_ids = wrap_with_special_tokens(
                    self.tokenizer, self.query_dataset[negative_query_index]['input_ids'])

        else:
            positive_document_input_ids = None
            negative_document_input_ids = None

        return {
            'input_ids': query_input_ids,
            'positive_document_input_ids': positive_document_input_ids,
            'negative_document_input_ids': negative_document_input_ids,
            'labels': positive_document_index,
        }

    def __len__(self):
        return len(self.query_dataset)


@dataclass
class DataCollatorForBiEncoder:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        positive_document_input_ids = [
            f.get('positive_document_input_ids', None) for f in features]
        negative_document_input_ids = [
            f.get('negative_document_input_ids', None) for f in features]
        for f in features:
            if 'positive_document_input_ids' in f:
                del f['positive_document_input_ids']
            if 'negative_document_input_ids' in f:
                del f['negative_document_input_ids']

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        if positive_document_input_ids[0] is not None:
            batch['positive_document_input_ids'] = self.tokenizer.pad(
                {'input_ids': positive_document_input_ids},
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )['input_ids']
            batch['negative_document_input_ids'] = self.tokenizer.pad(
                {'input_ids': negative_document_input_ids},
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )['input_ids']

        return batch


class BiEncoderMethod(MethodWithDocumentDataset):
    name = "bi_encoder"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.metrics = [
            load_metric(metric) for metric in ['accuracy']
        ]
        self.config.encoder_shared = self.model_args.bi_encoder_shared
        self.config.triplet_loss_margin = self.model_args.triplet_loss_margin

    def preprocess_features(self, features):
        input_ids = [
            process_input(self.model_args, turns, self.tokenizer) for turns in features['turns']
        ]

        return {
            'input_ids': input_ids,
            'domain': [x[0]['domain'] for x in features['knowledge']],
            'entity_id': [x[0]['entity_id'] for x in features['knowledge']],
            'doc_id': [x[0]['doc_id'] for x in features['knowledge']],
        }

    def preprocess_documents(self, features, add_special_tokens=False):
        out = {
            'input_ids': process_knowledge(self.model_args, self.tokenizer, features),
            'domain': features['domain'],
        }
        if add_special_tokens:
            out['input_ids'] = wrap_with_special_tokens(
                self.tokenizer, out['input_ids'])
        if 'entity_id' in features:
            out['entity_id'] = features['entity_id']
        if 'doc_id' in features:
            out['doc_id'] = features['doc_id']
        return out

    def get_model_class(self, config: PretrainedConfig):
        return RobertaForBiEncoder

    def compute_metrics(self, p: EvalPrediction):
        margins = p.predictions[1]
        prediction_ids = margins < 0
        label_ids = [1] * len(margins)
        results = {}
        for metric in self.metrics:
            results.update(
                metric.compute(predictions=prediction_ids,
                               references=label_ids)
            )
        return results

    def postprocess_predictions(self, p: PredictionOutput, dataset):
        dataset.documents_dataset.load_faiss_index(
            'embeddings', self.data_args.test_documents_faiss_index_path)

        def get_documents(query_idx):
            query_embedding = p.predictions[0][query_idx]
            scores, indices = dataset.documents_dataset.search('embeddings', query_embedding,
                                                               self.model_args.selection_prediction_topk)

            data = []
            for score, index in zip(scores, indices):
                sample = {
                    'score': score.item(),
                    **{
                        k: v for k, v in dataset.documents_dataset[index.item()].items()
                        if k in ['domain', 'entity_id', 'doc_id']
                    },
                }
                sample["body"] = self.tokenizer.batch_decode([dataset.documents_dataset[index.item()]["input_ids"]])[0].split("<knowledge_sep>")[-1].strip()
                data.append(sample)
            return data

        return list(map(get_documents, range(len(dataset.query_dataset))))

    def _get_dataset(self, split, config_name=None):
        assert config_name is None
        query_dataset = super()._get_dataset(split)

        # Remove the slice when loading the document dataset
        document_dataset = self.get_document_dataset(split)
        document_register = build_knowledge_document_register(document_dataset)
        old_eval_column_names = document_dataset.column_names
        document_dataset = document_dataset.map(
            self.preprocess_documents,
            batched=False,
            remove_columns=old_eval_column_names,
        )

        return BiEncoderDataset(
            self.model_args,
            self.tokenizer,
            query_dataset,
            document_dataset,
            document_register,
            self.data_args.is_training,
        )

    def get_data_collator(self):
        return DataCollatorForBiEncoder(self.tokenizer)
