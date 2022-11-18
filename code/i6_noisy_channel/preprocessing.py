import itertools
import random

from i6_noisy_channel.arguments import ModelArguments


def truncate_sequences(sequences, max_length):
  words_to_cut = sum(list(map(len, sequences))) - max_length
  if words_to_cut <= 0:
    return sequences

  while words_to_cut > len(sequences[0]):
    words_to_cut -= len(sequences[0])
    sequences = sequences[1:]

  sequences[0] = sequences[0][words_to_cut:]
  return sequences


def process_input(args: ModelArguments, turns, tokenizer):
  history = [
    tokenizer.convert_tokens_to_ids(tokenizer.tokenize(turn["text"]))
    for turn in turns
  ]

  # apply history threshold at an utterance-level (a large value can be used to nullify its effect)
  truncated_history = history[-args.history_max_utterances:]

  # perform token-level truncation of history from the left
  truncated_history = truncate_sequences(truncated_history, args.history_max_tokens)
  truncated_speaker = [turn['speaker'] for turn in turns[-len(truncated_history):]]

  # Add speaker tags to the history and response
  # the response is always by speaker2
  # and the current_turn always by speaker1
  truncated_history = [
    tokenizer.convert_tokens_to_ids(tokenizer.tokenize(args.user_token if s == 'U' else args.agent_token)) + t
    for i, t, s in zip(range(len(truncated_history)), truncated_history, truncated_speaker)
  ]

  return list(itertools.chain.from_iterable(truncated_history))


def create_concatenated_model_input(args: ModelArguments, turns, tokenizer, knowledge=None):
  input_ids = process_input(args, turns, tokenizer)

  if knowledge is not None:
    truncated_knowledge = process_knowledge(args, tokenizer, knowledge)

  if knowledge is None:
    input_ids = wrap_with_special_tokens(tokenizer, input_ids)
  else:
    input_ids = create_concatenated_dialog_knowledge_input(args, tokenizer, input_ids, truncated_knowledge)

  return input_ids


def wrap_with_special_tokens(tokenizer, input_ids):
  return [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]


def create_concatenated_dialog_knowledge_input(args: ModelArguments, tokenizer, dialog_input_ids, knowledge_input_ids):
  return [tokenizer.cls_token_id] + knowledge_input_ids + tokenizer.convert_tokens_to_ids(
      [args.knowledge_tag_token]) + dialog_input_ids + [tokenizer.sep_token_id]


def create_input_for_reranking(args: ModelArguments, tokenizer, dialog_input_ids, knowledge_input_ids_list):
  input_ids = []
  knowledge_sep = tokenizer.convert_tokens_to_ids([args.knowledge_tag_token])
  for knowledge_input_ids in knowledge_input_ids_list:
    input_ids += knowledge_sep + knowledge_input_ids
  max_length = 448
  assert len(input_ids) < max_length - 128
  input_ids = dialog_input_ids[max(0, len(input_ids) + len(dialog_input_ids) - max_length):] + input_ids
  assert len(input_ids) <= max_length
  start_indices = [i for i, token in enumerate(input_ids) if token == knowledge_sep[0]]
  if args.selection_reranking_include_non_ks:
    start_indices = [0] + start_indices
  return wrap_with_special_tokens(tokenizer, input_ids), start_indices


def prepare_knowledge(args: ModelArguments, knowledge):
  snippet = knowledge
  join_str = " %s " % args.knowledge_sep_token

  if args.selection_level == 'domain':
    knowledge_parts = [snippet["domain"]]
  elif args.selection_level in ['entity', 'domain_entity']:
    knowledge_parts = [snippet["domain"], snippet['entity_name']]
  else:
    knowledge_parts = [snippet["domain"], snippet['entity_name'], snippet['title'], snippet['body']]

  knowledge_to_use = join_str.join([k for k in knowledge_parts if k is not None])

  return knowledge_to_use


def process_knowledge(args: ModelArguments, tokenizer, knowledge):
  knowledge_text = prepare_knowledge(args, knowledge)
  tokenized_knowledge = tokenizer(knowledge_text, add_special_tokens=False)["input_ids"]
  truncated_knowledge = tokenized_knowledge[:args.knowledge_max_tokens]
  return truncated_knowledge
