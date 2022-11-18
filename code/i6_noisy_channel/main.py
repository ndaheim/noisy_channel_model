import dataclasses
import itertools
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import bert_score
import numpy as np
import torch
from datasets import load_dataset, load_metric, Dataset
from sacrebleu import BLEU

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    default_data_collator,
    set_seed, Trainer, Seq2SeqTrainingArguments, TrainerCallback, TrainerState, TrainerControl,
)
from transformers.trainer_utils import is_main_process, PredictionOutput, get_last_checkpoint

from i6_noisy_channel.arguments import *
from i6_noisy_channel.methods.base import Method, MethodWithDocumentDataset
from i6_noisy_channel.methods.bi_encoder import BiEncoderMethod
from i6_noisy_channel.methods.cross_encoder import CrossEncoderMethod
from i6_noisy_channel.methods.generation import DocumentGroundedGenerationMethod, ResponseGenerationMethod, CTRLMethod
from i6_noisy_channel.methods.noisy_channel import NoisyChannelOnlineWithCTRLTokensMethod, CTRLChannelModelMethod, CTRLOnlineChannelModelMethod, ChannelModelMethod, NoisyChannelModelMethod, OnlineChannelModelMethod, GenerationWithOnlineNoisyChannelMethod, NoisyChannelOnlineLiuEtAl, NoisyChannelWithCTRLTokensMethod
from i6_noisy_channel.models.noisy_channel import NoisyChannelConfig
from i6_noisy_channel.utils import NumpyEncoder

logger = logging.getLogger(__name__)
os.environ["WANDB_DISABLED"] = "true"

method_classes = [
    ResponseGenerationMethod,
    DocumentGroundedGenerationMethod,
    CrossEncoderMethod,
    CTRLMethod,
    BiEncoderMethod,
    CTRLChannelModelMethod,
    CTRLOnlineChannelModelMethod,
    ChannelModelMethod,
    NoisyChannelModelMethod,
    OnlineChannelModelMethod,
    GenerationWithOnlineNoisyChannelMethod,
    NoisyChannelOnlineLiuEtAl,
    NoisyChannelWithCTRLTokensMethod,
    NoisyChannelOnlineWithCTRLTokensMethod
]


class GPUMemoryCallback(TrainerCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prediction_step = 0

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if torch.cuda.is_available():
            max_gpu_allocated = torch.cuda.max_memory_allocated() / 10**9
            print(f"Maximum allocated GPU memory: {max_gpu_allocated:.3f} GB")
            state.log_history[-1]['gpu_memory'] = torch.cuda.max_memory_allocated()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step in [1, 8, 64, 512]:
            if torch.cuda.is_available():
                max_gpu_allocated = torch.cuda.max_memory_allocated() / 10**9
                print(
                    f"Maximum allocated GPU memory: {max_gpu_allocated:.3f} GB")
        super().on_step_end(args, state, control, **kwargs)

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.prediction_step += 1
        if self.prediction_step in [1, 8, 64, 512]:
            if torch.cuda.is_available():
                max_gpu_allocated = torch.cuda.max_memory_allocated() / 10**9
                print(
                    f"Maximum allocated GPU memory: {max_gpu_allocated:.3f} GB")
        super().on_prediction_step(args, state, control, **kwargs)


def _setup_logging(training_args: TrainingArguments):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    logger.setLevel(logging.INFO if is_main_process(
        training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)


def get_config_class(model_args):
    if model_args.method in ["noisy_channel", "noisy_channel_online", "noisy_channel_liuetal", "noisy_channel_reranking", "noisy_channel_reranking_with_ctrl_tokens", "noisy_channel_online_with_ctrl_tokens"]:
        return NoisyChannelConfig
    else:
        return AutoConfig


def get_tokenizer_name(config, model_args):
    if model_args.method in ["noisy_channel", "noisy_channel_online", "noisy_channel_reranking", "noisy_channel_liuetal", "noisy_channel_reranking_with_ctrl_tokens", "noisy_channel_online_with_ctrl_tokens"]:
        return config.direct_model_tokenizer_name_or_path
    elif model_args.tokenizer_name:
        return model_args.tokenizer_name
    else:
        return model_args.model_name_or_path


class RunMode(Enum):
    TRAIN = 1
    PREDICT = 2
    BUILD_INDEX = 3
    PERPLEXITY = 4
    SEARCH_ERRORS = 5


def main(run_mode: RunMode):
    training_args_class = Seq2SeqTrainingArguments
    parser_arguments = (ModelArguments, DataTrainingArguments if run_mode ==
                        RunMode.TRAIN else DataPredictionArguments, training_args_class)
    parser = HfArgumentParser(parser_arguments)

    raw_args = sys.argv[1:]
    json_index = -1 if raw_args[-1].endswith(".json") and (len(
        raw_args) == 1 or not raw_args[-2].startswith('-') or '=' in raw_args[-2]) else 0
    if len(raw_args) > 0 and raw_args[json_index].endswith(".json"):
        with open(raw_args[json_index]) as fp:
            json_args_dict = json.load(fp)
        del raw_args[json_index]

        if run_mode == RunMode.TRAIN:
            train_parser = HfArgumentParser(training_args_class)
            training_args_dict = vars(train_parser.parse_args(
                raw_args + ['--output_dir', json_args_dict['output_dir']]))
            training_args_dict.update(json_args_dict)
            json_args_dict = training_args_dict

        model_args, data_args, training_args = parser.parse_dict(
            json_args_dict)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print(data_args)

    print(
        f"My rank is {training_args.local_rank} with {torch.cuda.device_count()} GPUs.")
    if training_args.local_rank != -1:
        torch.cuda.set_device(training_args.local_rank)

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    _setup_logging(training_args)

    config_class = get_config_class(model_args)

    config = config_class.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        get_tokenizer_name(config, model_args),
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if model_args.method == 'bi_encoder':
        config.model_parallel_encoders = torch.cuda.device_count(
        ) == 2 and training_args.local_rank == -1 and not model_args.bi_encoder_shared

    method_class = next(
        (m for m in method_classes if m.name == model_args.method), None)
    if method_class is None:
        raise Exception(f"No method class for name {model_args.method}.")
    method_definition: Method = method_class(
        model_args, data_args, config, tokenizer)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    model = method_definition.get_model(run_mode, config)
    model.config.num_beams = model_args.generation_beam_size
    model.config.max_length = model_args.generation_max_len
    model.config.generation_typical_p = model_args.generation_typical_p
    model.config.do_sample = model_args.generation_do_sample
    model.config.length_penalty = model_args.generation_length_penalty
    model.config.no_repeat_ngram_size = model_args.generation_no_repeat_ngram_size
    model.config.uid_regularization = model_args.generation_uid_regularization
    model.config.k_2 = model_args.generation_k_2

    if run_mode == RunMode.TRAIN:
        extra_trainer_args = {
            'train_dataset': method_definition.get_train_dataset(),
            'eval_dataset': method_definition.get_eval_dataset(),
        }
    else:
        extra_trainer_args = {}

    data_collator = method_definition.get_data_collator()
    trainer_class = method_definition.get_trainer_class()

    trainer: Trainer = trainer_class(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=method_definition.compute_metrics,
        **extra_trainer_args,
    )
    trainer.add_callback(GPUMemoryCallback())

    if run_mode == RunMode.TRAIN:
        # Check for existing checkpoint to continue the training
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        resume_from_checkpoint = last_checkpoint if last_checkpoint is not None else None
        # Start training
        train_result = trainer.train(
            resume_from_checkpoint=resume_from_checkpoint)

        output_train_file = os.path.join(
            training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(
                training_args.output_dir, "trainer_state.json"))

    elif run_mode in [RunMode.PREDICT]:
        test_dataset = method_definition.get_test_dataset()
        results = trainer.predict(test_dataset)

        if run_mode == RunMode.PREDICT:
            results = method_definition.postprocess_predictions(
                results, test_dataset)

        if data_args.prediction_output_file is not None:
            with open(data_args.prediction_output_file, 'wt') as fp:
                json.dump(
                    dataclasses.asdict(results) if type(
                        results) == PredictionOutput else results,
                    fp, cls=NumpyEncoder
                )

        if data_args.dataset_config_name in ["generation", "control"]:
            test_dataset = method_definition.get_test_dataset(evaluation=True)

            refs = [[sample["response"] for sample in test_dataset]]

            bleu = BLEU()
            results = [sample["response"] for sample in results if sample["target"]]
            score = bleu.corpus_score(results, refs)

            refs = list(itertools.chain.from_iterable(refs))
            bert_scores = bert_score.score(results, refs, model_type="bert-base-multilingual-cased")
            P, R, f1 = (tensor.mean().item() for tensor in bert_scores)
            scores = {
                "sacrebleu": score.score,
                "sacrebleu_signature": str(bleu.get_signature()),
                "BertScore": {
                    "P": P,
                    "R": R,
                    "F-1": f1
                }
            }

            if data_args.metric_output_file is not None:
                with open(data_args.metric_output_file, "wt") as f:
                    json.dump(
                        scores,
                        f
                    )

    elif run_mode == RunMode.BUILD_INDEX:
        import faiss
        assert isinstance(method_definition, MethodWithDocumentDataset)
        assert data_args.test_documents_faiss_index_path is not None

        document_dataset = method_definition.get_document_dataset(
            data_args.dataset_test_split)
        old_eval_column_names = document_dataset.column_names
        document_dataset = document_dataset.map(
            method_definition.preprocess_documents,
            batched=False,
            remove_columns=old_eval_column_names,
            fn_kwargs={'add_special_tokens': True},
        )

        if model_args.bi_encoder_loss:
            metric_type = faiss.METRIC_L2
        else:
            raise Exception("Loss not implemented")

        embeddings = trainer.predict(document_dataset).predictions[0]

        document_dataset.add_faiss_index_from_external_arrays(
            embeddings, 'embeddings', metric_type=metric_type)
        document_dataset.save_faiss_index(
            'embeddings', data_args.test_documents_faiss_index_path)
