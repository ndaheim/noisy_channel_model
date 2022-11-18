from collections import defaultdict

from datasets import Dataset

from i6_noisy_channel.utils import defaultdict2dict


def build_knowledge_document_register(document_dataset: Dataset):
  """
  Creates a register which maps the domain, entity_id and doc_id of a document to the index in the document_dataset.

  :param document_dataset:
  :return: dict with keys [domain][entity_id][doc_id] -> index
  """
  if 'doc_id' in document_dataset.column_names:
    register = defaultdict(lambda: defaultdict(dict))
    for idx, doc in enumerate(document_dataset):
      register[doc['domain']][doc['entity_id']][doc['doc_id']] = idx
  elif 'entity_id' in document_dataset.column_names:
    register = defaultdict(dict)
    for idx, doc in enumerate(document_dataset):
      register[doc['domain']][doc['entity_id']] = idx
  else:
    register = dict()
    for idx, doc in enumerate(document_dataset):
      register[doc['domain']] = idx

  return defaultdict2dict(register)


def build_knowledge_document_register_with_city(document_dataset: Dataset):
  """
  Creates a register which maps the domain, entity_id and doc_id of a document to the index in the document_dataset.

  :param document_dataset:
  :return: dict with keys [domain][entity_id][doc_id] -> index
  """
  assert 'city' in document_dataset.column_names
  if 'doc_id' in document_dataset.column_names:
    register = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for idx, doc in enumerate(document_dataset):
      register[doc['city']][doc['domain']][doc['entity_id']][doc['doc_id']] = idx
      register['*'][doc['domain']][doc['entity_id']][doc['doc_id']] = idx
  elif 'entity_id' in document_dataset.column_names:
    register = defaultdict(lambda: defaultdict(dict))
    for idx, doc in enumerate(document_dataset):
      register[doc['city']][doc['domain']][doc['entity_id']] = idx
      register['*'][doc['domain']][doc['entity_id']] = idx
  else:
    register = defaultdict(dict)
    for idx, doc in enumerate(document_dataset):
      register[doc['city']][doc['domain']] = idx
      register['*'][doc['domain']] = idx

  return defaultdict2dict(register)
