import itertools
import json

import numpy as np
from datasets import load_metric, load_dataset, concatenate_datasets, Dataset, load_from_disk
from transformers import PretrainedConfig, EvalPrediction, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, AutoTokenizer
from transformers.trainer_utils import PredictionOutput

from i6_noisy_channel.methods.base import Method
from i6_noisy_channel.methods.generation import DocumentGroundedGenerationMethod
from i6_noisy_channel.methods.trainer_seq2seq import CustomSeq2SeqTrainer, DataCollatorForSeq2SeqForPartialInputs, CustomSeq2SeqTrainerWithPartialInputs
from i6_noisy_channel.models.noisy_channel import NoisyChannelRerankingModelForConditionalGeneration, OnlineNoisyChannelModelForConditionalGeneration, OnlineNoisyChannelModelLiuEtAl
from i6_noisy_channel.preprocessing import create_concatenated_model_input, process_knowledge


class ChannelModelMethod(DocumentGroundedGenerationMethod):

    name = "channel_model"

    def preprocess_features(self, features):
        features["response"] =  [[{"speaker": "S", "text": response}] for response in features["response"]]
        input_ids = [
        create_concatenated_model_input(self.model_args, turns + response, self.tokenizer, knowledge=None)
            for turns, response in zip(features["turns"], features['response'])
        ]
        target = [
        create_concatenated_model_input(self.model_args, [], self.tokenizer, knowledge=knowledge[0])
            for turns, knowledge in zip(features['turns'], features['knowledge'])
        ]

        return_dict = {
            'input_ids': input_ids,
            'labels': target
        }

        return return_dict


class OnlineChannelModelMethod(DocumentGroundedGenerationMethod):

    name = "online_channel_model"

    def get_data_collator(self):
        return DataCollatorForSeq2SeqForPartialInputs(self.tokenizer)

    def get_trainer_class(self):
        return CustomSeq2SeqTrainerWithPartialInputs

    def preprocess_features(self, features):
        features["response"] =  [[{"speaker": "S", "text": response}] for response in features["response"]]
        input_ids = [
        create_concatenated_model_input(self.model_args, turns + response, self.tokenizer, knowledge=None)
            for turns, response in zip(features["turns"], features['response'])
        ]
        target = [
        create_concatenated_model_input(self.model_args, [], self.tokenizer, knowledge=knowledge[0])
            for turns, knowledge in zip(features['turns'], features['knowledge'])
        ]
        
        return_dict = {
            'input_ids': input_ids, 
            'labels': target 
        }

        return return_dict

        OnlineNoisyChannelModelForConditionalGeneration


class NoisyChannelModelMethod(DocumentGroundedGenerationMethod):

    name = "noisy_channel_reranking"

    def get_model_class(self, config: PretrainedConfig):
        return NoisyChannelRerankingModelForConditionalGeneration

    def preprocess_features(self, features):
        return_dict = super().preprocess_features(features)
        return_dict["lm_input_ids"] = [
            create_concatenated_model_input(self.model_args, turns, self.tokenizer, knowledge=None)
            for turns in features['turns']
        ]

        return_dict["cm_labels"] = [
            create_concatenated_model_input(self.model_args, [], self.tokenizer, knowledge=knowledge[0])
            for turns, knowledge in zip(features['turns'], features['knowledge'])
        ]

        return return_dict


class GenerationWithOnlineNoisyChannelMethod(DocumentGroundedGenerationMethod):

    name = "noisy_channel_online"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cm_tokenizer = AutoTokenizer.from_pretrained(self.config.channel_model_tokenizer_name_or_path)
        self.lm_tokenizer = AutoTokenizer.from_pretrained(self.config.language_model_tokenizer_name_or_path)

    def get_special_tokens(self):
        return [
            self.model_args.user_token,
            self.model_args.agent_token,
            self.model_args.knowledge_tag_token,
            self.model_args.knowledge_sep_token,
        ]

    def get_model_class(self, config: PretrainedConfig):
        return OnlineNoisyChannelModelForConditionalGeneration

    def preprocess_features(self, features):
        input_ids = [
            self.tokenizer.convert_tokens_to_ids([self.tokenizer.bos_token]) + create_concatenated_model_input(self.model_args, turns, self.tokenizer, knowledge=knowledge[0])[1:]
            for turns, knowledge in zip(features['turns'], features['knowledge'])
        ]

        return_dict = {
            'input_ids': input_ids,
        }

        if self.data_args.is_training:
            return_dict["labels"] = self.tokenizer(features["response"])["input_ids"]

        return_dict["lm_input_ids"] = [
            create_concatenated_model_input(self.model_args, turns, self.lm_tokenizer, knowledge=None)
            for turns in features['turns']
        ]

        return_dict["cm_labels"] = [
            create_concatenated_model_input(self.model_args, [], self.cm_tokenizer, knowledge=knowledge[0])
            for turns, knowledge in zip(features['turns'], features['knowledge'])
        ]

        return return_dict
        
    def get_model(self, run_mode, config: PretrainedConfig):
        model_class = self.get_model_class(config)
        model = model_class.from_pretrained(
        self.model_args.model_name_or_path,
        from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
        config=config,
        cache_dir=self.model_args.cache_dir,
        revision=self.model_args.model_revision,
        use_auth_token=True if self.model_args.use_auth_token else None,
        )
        return model


class NoisyChannelOnlineLiuEtAl(GenerationWithOnlineNoisyChannelMethod):

    name = "noisy_channel_liuetal"

    def get_model_class(self, config: PretrainedConfig):
        return OnlineNoisyChannelModelLiuEtAl


class CTRLChannelModelMethod(ChannelModelMethod):

    name = "ctrl_channel_model"
    
    def get_special_tokens(self):
        standard_tokens = super().get_special_tokens()
        entailement_tokens = ["<entailed>", "<non-entailed>"]
        lexical_tokens = ["<low-prec>", "<med-prec>", "<high-prec>"]
        special_tokens = standard_tokens + entailement_tokens + lexical_tokens
        return special_tokens

    def preprocess_features(self, features):
        return_dict = super().preprocess_features(features)
        
        for i, (ids, ctrl_tokens) in enumerate(zip(return_dict["input_ids"], features["control_tokens"])):
            ctrl_tokens = ctrl_tokens.replace("<no-first-person>", "").replace("<first-person>", "")
            ctrl_tokens = self.tokenizer.convert_tokens_to_ids(ctrl_tokens.strip().split(" "))
            return_dict["input_ids"][i] = return_dict["input_ids"][i][:1] + ctrl_tokens + return_dict["input_ids"][i][1:]

        return return_dict
    
class CTRLOnlineChannelModelMethod(OnlineChannelModelMethod):

    name = "ctrl_online_channel_model"
    
    def get_special_tokens(self):
        standard_tokens = super().get_special_tokens()
        entailement_tokens = ["<entailed>", "<non-entailed>"]
        lexical_tokens = ["<low-prec>", "<med-prec>", "<high-prec>"]
        special_tokens = standard_tokens + entailement_tokens + lexical_tokens
        return special_tokens

    def preprocess_features(self, features):
        return_dict = super().preprocess_features(features)
        
        for i, (ids, ctrl_tokens) in enumerate(zip(return_dict["input_ids"], features["control_tokens"])):
            ctrl_tokens = ctrl_tokens.replace("<no-first-person>", "").replace("<first-person>", "")
            ctrl_tokens = self.tokenizer.convert_tokens_to_ids(ctrl_tokens.strip().split(" "))
            return_dict["input_ids"][i] = return_dict["input_ids"][i][:1] + ctrl_tokens + return_dict["input_ids"][i][1:]

        return return_dict

class NoisyChannelWithCTRLTokensMethod(DocumentGroundedGenerationMethod):
    
    name = "noisy_channel_reranking_with_ctrl_tokens"
    
    def get_model_class(self, config: PretrainedConfig):
        return NoisyChannelRerankingModelForConditionalGeneration
    
    def preprocess_features(self, features):
        return_dict = super().preprocess_features(features)
        lm_tokenizer = AutoTokenizer.from_pretrained(self.config.language_model_tokenizer_name_or_path)
        return_dict["lm_input_ids"] = [
            create_concatenated_model_input(self.model_args, turns, lm_tokenizer, knowledge=None)
            for turns in features['turns']
        ]

        return_dict["cm_labels"] = [
            create_concatenated_model_input(self.model_args, [], self.tokenizer, knowledge=knowledge[0])
            for turns, knowledge in zip(features['turns'], features['knowledge'])
        ]
        
        cm_tokenizer = AutoTokenizer.from_pretrained(self.config.channel_model_tokenizer_name_or_path)
        
        return_dict["cm_input_ids"] = [
            create_concatenated_model_input(self.model_args, turns, cm_tokenizer, knowledge=None)
            for turns in features['turns']
        ]
        
        ctrl_tokens = ["<high-prec>", "<entailed>"]
        ctrl_tokens = self.tokenizer.convert_tokens_to_ids(ctrl_tokens)
        for i, ids in enumerate(return_dict["input_ids"]):
            return_dict["input_ids"][i] = return_dict["input_ids"][i][:1] + ctrl_tokens + return_dict["input_ids"][i][1:]
          
        ctrl_tokens = ["<high-prec>", "<entailed>"]
        ctrl_tokens = cm_tokenizer.convert_tokens_to_ids(ctrl_tokens)  
        for i, ids in enumerate(return_dict["cm_input_ids"]):
            return_dict["cm_input_ids"][i] = return_dict["cm_input_ids"][i][:1] + ctrl_tokens + return_dict["cm_input_ids"][i][1:]

        return return_dict
    
    def get_model(self, run_mode, config: PretrainedConfig):
        model_class = self.get_model_class(config)
        model = model_class.from_pretrained(
            self.model_args.model_name_or_path,
            from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
            config=config,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
        )
        model.direct_model.resize_token_embeddings(len(self.tokenizer))
        model.channel_model.resize_token_embeddings(len(self.tokenizer))
        return model

class NoisyChannelOnlineWithCTRLTokensMethod(DocumentGroundedGenerationMethod):
    
    name = "noisy_channel_online_with_ctrl_tokens"
    
    def get_model_class(self, config: PretrainedConfig):
        return OnlineNoisyChannelModelForConditionalGeneration
    
    def preprocess_features(self, features):
        return_dict = super().preprocess_features(features)
        lm_tokenizer = AutoTokenizer.from_pretrained(self.config.language_model_tokenizer_name_or_path)
        return_dict["lm_input_ids"] = [
            create_concatenated_model_input(self.model_args, turns, lm_tokenizer, knowledge=None)
            for turns in features['turns']
        ]

        return_dict["cm_labels"] = [
            create_concatenated_model_input(self.model_args, [], self.tokenizer, knowledge=knowledge[0])
            for turns, knowledge in zip(features['turns'], features['knowledge'])
        ]
        
        cm_tokenizer = AutoTokenizer.from_pretrained(self.config.channel_model_tokenizer_name_or_path)
        
        return_dict["cm_input_ids"] = [
            create_concatenated_model_input(self.model_args, turns, cm_tokenizer, knowledge=None)
            for turns in features['turns']
        ]
        
        ctrl_tokens = ["<high-prec>", "<entailed>"]
        ctrl_tokens = self.tokenizer.convert_tokens_to_ids(ctrl_tokens)
        for i, ids in enumerate(return_dict["input_ids"]):
            return_dict["input_ids"][i] = return_dict["input_ids"][i][:1] + ctrl_tokens + return_dict["input_ids"][i][1:]
          
        ctrl_tokens = ["<high-prec>", "<entailed>"]
        ctrl_tokens = cm_tokenizer.convert_tokens_to_ids(ctrl_tokens)  
        for i, ids in enumerate(return_dict["cm_input_ids"]):
            return_dict["cm_input_ids"][i] = return_dict["cm_input_ids"][i][:1] + ctrl_tokens + return_dict["cm_input_ids"][i][1:]

        return return_dict
    
    def get_model(self, run_mode, config: PretrainedConfig):
        model_class = self.get_model_class(config)
        model = model_class.from_pretrained(
            self.model_args.model_name_or_path,
            from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
            config=config,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
        )
        model.direct_model.resize_token_embeddings(len(self.tokenizer))
        model.channel_model.resize_token_embeddings(len(self.tokenizer))
        return model