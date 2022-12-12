# Noisy Channel Model For Document-Grounded Dialog

This repository contains the implementation of the paper `Controllable Factuality in Document-Grounded Dialog Systems Using A Noisy Channel Model`, EMNLP Findings 2022

The repository is structured as follows:
- `/code` contains all our code and is divided as follows:
    - `/datasets` contains dataloaders for [Huggingface datasets](https://github.com/huggingface/datasets) and code for data augmentation
    - `/methods` contains the definitions of our *methods* which define the model and preprocessing that is used
    - `/models` contains implementations of models
- `/config` contains our [sisyphus](https://github.com/rwth-i6/sisyphus) setup which we used to run our experiments

## Running our code
The easiest way of running our code is to install [sisyphus](https://github.com/rwth-i6/sisyphus) and use our pre-written configs.

`/config/generation.py` contains baseline implementations for quickly training a baseline document-grounded response generation model, all noisy channel model components for reranking and online decoding and the CTRL model. Furthermore, it allows to run decoding using noisy channel reranking and both presented online decoding algorithms.

If you do not want to use sisyphus, it is also possible to run the code directly, by calling `code/train.py`and `code/predict.py` with a training or search config. In the following, we show two example configs, for all possible parameters, see `code/arguments.py`.

## Example config for training a document-grounded response generation model

```
{
    "predict_with_generate": true,
    "learning_rate": 6.25e-05,
    "generation_max_length": 60,
    "generation_beam_size": 10,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 32,
    "model_name_or_path": "facebook/bart-base",
    "method": "document_grounded_generation",
    "evaluation_strategy": "epoch",
    "output_dir": $OUTPUT_DIR,
    "num_train_epochs": 10,
    "logging_strategy": "steps",
    "logging_steps": 128,
    "save_strategy": "epoch",
    "overwrite_output_dir": true,
    "dataset_config_name": "generation",
    "dataset_train_split": "train",
    "dataset_eval_split": "validation",
    "dataset_name": "{$CODE_ROOT}/code/i6_noisy_channel/datasets/{$DATASET}.py"
}
```

## Example config for decoding a document-grounded response generation model

```
{
    "predict_with_generate": true,
    "learning_rate": 6.25e-05,
    "generation_max_length": 60,
    "generation_beam_size": 10,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 32,
    "model_name_or_path": $MODEL_PATH,
    "method": "document_grounded_generation",
    "metric_output_file": $OUTPUT_METRICS,
    "prediction_output_file": $OUTPUT_PREDICTIONS,
    "output_dir": "trainer_output_dir",
    "config_name": null,
    "tokenizer_name": null,
    "dataset_config_name": "generation",
    "dataset_test_split": "test",
    "dataset_name": "{$CODE_ROOT}/code/i6_noisy_channel/datasets/{$DATASET}.py"
}
```


For using the noisy channel model, `model_name_or_path` has to point to a checkpoint created using `code/i6_noisy_channel/models/create_nc_checkpoint.py`that wraps all components into one model.
## Citation

If you use part of this work, please cite [our paper](https://arxiv.org/pdf/2210.17418.pdf):

```
@inproceedings{daheimNoisyChannelEMNLP2022,
  title = {Controllable {{Factuality in Document-Grounded Dialog Systems Using}} a {{Noisy Channel Model}}},
  booktitle = {{{{EMNLP Findings}}}},
  author = {Daheim, Nico and Thulke, David and Dugast, Christian and Ney, Hermann},
  publisher = {{Association for Computational Linguistics}},
  year = {2022},
}

```
