import logging

import datasets
import numpy as np
import spacy
import transformers
from accelerate import Accelerator
from datasets import Dataset
from nltk.tokenize import RegexpTokenizer
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from tqdm import tqdm


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
tokenizer = RegexpTokenizer(r'\w+')
nlp = spacy.load("en_core_web_sm", disable=("ner", "parser", "tagger", "lemmatizer"))

class CTRLMixin(object):
    
    _ENTAILMENT_TOKEN_MAP = {
        0: "<non-entailed>",
        1: "<non-entailed>",
        2: "<entailed>"
    }
    
    _LEXICAL_TOKEN_MAP = {
        0: "<low-prec>",
        1: "<med-prec>",
        2: "<high-prec>"
    }

    def _tokenize(self, text, as_string=False):
        # tokens = [tok.text for tok in nlp(text)]
        tokens = tokenizer.tokenize(text)
        if as_string:
            return " ".join(tokens)
        else:
            return tokens
        
    def _compute_lexical_overlap_group(self, lexical_overlaps):
        lexical_overlaps = np.array(lexical_overlaps)
        sorted_lex_indices = np.argsort(lexical_overlaps)
        lo_indices, med_indices, _ = np.array_split(sorted_lex_indices, 3)
        max_lo_overlap = lexical_overlaps[lo_indices[-1]] if lo_indices.size > 0 else 0
        max_med_overlap = lexical_overlaps[med_indices[-1]] if med_indices.size > 0 else 0

        groups = [-1] * len(lexical_overlaps)

        for idx in range(len(lexical_overlaps)):
            if lexical_overlaps[idx] <= max_lo_overlap:
                groups[idx] = 0
            elif max_lo_overlap < lexical_overlaps[idx] <= max_med_overlap:
                groups[idx] = 1
            else:
                groups[idx] = 2

        return groups
        
    def _measure_lexical_overlap(self, tokens, ref_tokens):
        """
        Noted in https://aclanthology.org/2021.acl-long.58/:
        "this may not reflect semantic differences in the information being shared
        (e.g. dropping the word ‘not’ may yield high lexical precision but a very different semantic meaning
        from the original evidence)."
        :param tokens: utterance tokens
        :param ref_tokens: reference tokens
        :return: lexical overlap, ratio of common terms over length of tokens
        """
        if not tokens:
            return 0.0

        return sum(1 for t in tokens if t in ref_tokens) / len(tokens)
    
    def _predict_nli_labels(self, model_name_or_path, nli_data, max_length=384, per_device_batch_size=2):
        accelerator = Accelerator()
        logger.info(accelerator.state)
        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.logging.set_verbosity_error()

        config = AutoConfig.from_pretrained(model_name_or_path, num_labels=3, finetuning_task="mnli")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
        )

        def preprocess_function(examples):
            # Tokenize the texts
            texts = (examples["premise"], examples["hypothesis"])
            result = tokenizer(*texts, padding=False, max_length=max_length, truncation=True)

            return result

        raw_dataset = Dataset.from_dict(nli_data)
        processed_dataset = raw_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_dataset.column_names,
            desc="Running tokenizer on dataset",
        )

        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)
        dataloader = DataLoader(processed_dataset, collate_fn=data_collator, batch_size=per_device_batch_size)
        model, dataloader = accelerator.prepare(model, dataloader)
        model.eval()
        for step, batch in enumerate(tqdm(dataloader, total=len(processed_dataset))):
            outputs = model(**batch)
            predictions = accelerator.gather(outputs.logits.argmax(dim=-1)).detach().cpu().tolist()
            yield from predictions
        
    def _add_control_tokens(self, data):
        nli_data = {
            "premise": [],
            "hypothesis": [],
            "did": [],
        }
        
        lexical_overlaps = []
        for idx, sample in tqdm(enumerate(data)):
            sample["control_tokens"] = ""
            premise = sample["knowledge"][0]["body"]
            knowledge_tokens = self._tokenize(premise)
            hypothesis = sample["response"]
            response_tokens = self._tokenize(hypothesis)
            
            nli_data["premise"].append(premise)
            nli_data["hypothesis"].append(hypothesis)
            nli_data["did"].append(idx)
            
            lexical_overlap = self._measure_lexical_overlap(response_tokens, knowledge_tokens)
            lexical_overlaps.append(lexical_overlap)
        lexical_overlaps = np.array(lexical_overlaps)
        lexical_groups = self._compute_lexical_overlap_group(lexical_overlaps)
        
        for sample, lexical_group in zip(data, lexical_groups):
            sample["control_tokens"] += self._LEXICAL_TOKEN_MAP[lexical_group]
        
        nli_model = "roberta-large-mnli"
        nli_labels = list(self._predict_nli_labels(nli_model, nli_data))
        
        for sample, nli_label in zip(data, nli_labels):
            sample["control_tokens"] += f" {self._ENTAILMENT_TOKEN_MAP[nli_label]}"
        return data