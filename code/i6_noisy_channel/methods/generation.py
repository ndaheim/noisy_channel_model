import itertools
import json

from datasets import load_dataset, concatenate_datasets
from evaluate import load
from transformers import PretrainedConfig, EvalPrediction, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, AutoTokenizer
from transformers.trainer_utils import PredictionOutput

from i6_noisy_channel.methods.base import Method
from i6_noisy_channel.methods.trainer_seq2seq import CustomSeq2SeqTrainer, DataCollatorForSeq2SeqWithLMInputs
from i6_noisy_channel.preprocessing import create_concatenated_model_input


class DocumentGroundedGenerationMethod(Method):
    name = "document_grounded_generation"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.metrics = [
            load(metric) for metric in ['sacrebleu']
        ]

    def preprocess_features(self, features):
        input_ids = [
            create_concatenated_model_input(
                self.model_args, turns, self.tokenizer, knowledge=knowledge[0])
            for turns, knowledge in zip(features['turns'], features['knowledge'])
        ]

        return_dict = {
            'input_ids': input_ids,
            'id': features['id'],
            'target': features['target']
        }

        if self.data_args.is_training:
            targets = self.tokenizer(features["response"])["input_ids"]
            for i, target in enumerate(targets):
                if len(target) > 256:
                    targets[i] = target[:256]
            return_dict["labels"] = targets

        return return_dict

    def get_model_class(self, config: PretrainedConfig):
        return AutoModelForSeq2SeqLM

    def get_trainer_class(self):
        return CustomSeq2SeqTrainer

    def get_data_collator(self):
        return DataCollatorForSeq2SeqWithLMInputs(self.tokenizer)

    def compute_metrics(self, p: EvalPrediction):
        predictions_strings = self.tokenizer.batch_decode(
            p.predictions, skip_special_tokens=True)
        label_ids = p.label_ids
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id
        reference_strings = [[ref] for ref in self.tokenizer.batch_decode(
            p.label_ids, skip_special_tokens=True)]
        results = {}
        for metric in self.metrics:
            results.update(
                metric.compute(predictions=predictions_strings,
                               references=reference_strings)
            )
        return results

    def postprocess_predictions(self, p: PredictionOutput, dataset):
        generations = self.tokenizer.batch_decode(
            p.predictions, skip_special_tokens=True)
        out = []
        generation_dataset = self.get_full_dataset(config_name="evaluation")

        idx = 0
        for i, sample in enumerate(generation_dataset):
            if sample["target"]:
                item = {"target": True}
                item["response"] = generations[idx]
                knowledge = sample["knowledge"]
                for key in ["title", "body"]:
                    for knowledge_item in knowledge:
                        knowledge_item.pop(key)
                item["knowledge"] = knowledge
                idx += 1
            else:
                item = {"target": False}
                
            out.append(item)

        return out


class ResponseGenerationMethod(DocumentGroundedGenerationMethod):
    name = "response_generation"

    def preprocess_features(self, features):
        input_ids = [
            create_concatenated_model_input(
                self.model_args, turns, self.tokenizer, knowledge=None)
            for turns, knowledge in zip(features['turns'], features['knowledge'])
        ]

        return_dict = {
            'input_ids': input_ids,
            'id': features['id'],
            'target': features['target']
        }

        if self.data_args.is_training:
            targets = self.tokenizer(features["response"])["input_ids"]
            for i, target in enumerate(targets):
                if len(target) > 256:
                    targets[i] = target[:256]
            return_dict["labels"] = targets

        return return_dict
    
class CTRLMethod(DocumentGroundedGenerationMethod):
    
    name = "ctrl"
    
    def get_special_tokens(self):
        standard_tokens = super().get_special_tokens()
        entailement_tokens = ["<entailed>", "<non-entailed>"]
        lexical_tokens = ["<low-prec>", "<med-prec>", "<high-prec>"]
        special_tokens = standard_tokens + entailement_tokens + lexical_tokens
        return special_tokens

    def preprocess_features(self, features):
        input_ids = [
            create_concatenated_model_input(
                self.model_args, turns, self.tokenizer, knowledge=knowledge[0])
            for turns, knowledge in zip(features['turns'], features['knowledge'])
        ]
        
        if self.data_args.is_training:
            for i, (ids, ctrl_tokens) in enumerate(zip(input_ids, features["control_tokens"])):
                # remove first-person token and insert CTRL tokens into input
                ctrl_tokens = ctrl_tokens.replace("<no-first-person>", "").replace("<first-person>", "")
                ctrl_tokens = self.tokenizer.convert_tokens_to_ids(ctrl_tokens.strip().split(" "))
                input_ids[i] = input_ids[i][:1] + ctrl_tokens + input_ids[i][1:]
        else:
            ctrl_tokens = ["<high-prec>", "<entailed>"]
            ctrl_tokens = self.tokenizer.convert_tokens_to_ids(ctrl_tokens)
            for i, ids in enumerate(input_ids):
                input_ids[i] = input_ids[i][:1] + ctrl_tokens + input_ids[i][1:]

        return_dict = {
            'input_ids': input_ids,
            'id': features['id'],
            'target': features['target']
        }

        if self.data_args.is_training:
            targets = self.tokenizer(features["response"])["input_ids"]
            for i, target in enumerate(targets):
                if len(target) > 256:
                    targets[i] = target[:256]
            return_dict["labels"] = targets

        return return_dict
