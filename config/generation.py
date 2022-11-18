import sys

import numpy as np

from i6_noisy_channel.huggingface.evaluation import CalculateDSTC10MetricsJob, CalculateQ2Job, CalculatePerplexityJob

sys.setrecursionlimit(2500)

# ------------------------------ Sisyphus -------------------------------------

import sisyphus.toolkit as tk
from i6_noisy_channel.huggingface.training import *
from i6_noisy_channel.huggingface.search import *

Path = tk.Path

# ------------------------------ Recipes --------------------------------------

async def baseline():
    code_root = None # TODO: add path to code/ directory

    config = {
        'predict_with_generate': True,
        'learning_rate': 6.25e-5,
        'generation_max_length': 60,
        'generation_beam_size': 10,
        'per_device_train_batch_size': 16,
        'per_device_eval_batch_size': 32,
    }
    train_data_config = {
        'dataset_config_name': 'generation',
        'dataset_train_split': 'train',
        'dataset_eval_split': 'validation',
    }
    search_data_config = {
        'dataset_config_name': 'generation',
        'dataset_test_split': 'test'
    }

    for dataset in ["dstc9_track1"]:
        config["model_name_or_path"] = "facebook/bart-base"

        dataset_name = os.path.join(code_root, f'i6_noisy_channel/datasets/{dataset}.py')
        for task_config in [train_data_config, search_data_config]:
            task_config["dataset_name"] = dataset_name

        method = "document_grounded_generation"
        config["method"] = method

        train_job_dm = HuggingfaceTrainingJob(
            code_root=code_root,
            config=config,
            train_data_config=train_data_config,
            num_epochs=1,
            mem_rqmt=12,
            time_rqmt=24,
            gpu_rqmt=1
        )
        train_job_dm.add_alias(f"baselines/train_job-{method}_{dataset}")
        tk.register_output(f"baselines/{method}-{dataset}-bart-base", train_job_dm.out_best_model)

        search_job = HuggingfaceSearchJob(
            code_root=code_root,
            model_path=train_job_dm.out_best_model,
            config=config,
            search_data_config=search_data_config,
            mem_rqmt=8,
        )
        tk.register_output(f'baselines/{method}_{dataset}_test.out.json', search_job.out_search_file)
        tk.register_output(f'baselines/{method}_{dataset}_test.metrics.json', search_job.out_metric_file)

        scoring_job = CalculateQ2Job(
            code_root,
            search_data_config["dataset_name"],
            search_data_config['dataset_test_split'],
            search_job.out_search_file,
            time_rqmt=15
        )
        tk.register_output(f"baselines/{method}_{dataset}_test.q2.json", scoring_job.out_results_file)
        
        method = "channel_model"
        config["method"] = method

        train_job_cm = HuggingfaceTrainingJob(
            code_root=code_root,
            config=config,
            train_data_config=train_data_config,
            num_epochs=1,
            mem_rqmt=18,
            time_rqmt=24,
        )
        train_job_cm.add_alias(f"train_job-{method}_{dataset}")
        tk.register_output(f"{method}-{dataset}-bart-base", train_job_cm.out_best_model)

        method = "response_generation"
        config["method"] = method

        train_job_lm = HuggingfaceTrainingJob(
            code_root=code_root,
            config=config,
            train_data_config=train_data_config,
            num_epochs=1,
            mem_rqmt=18,
            time_rqmt=24,
        )
        train_job_lm.add_alias(f"train_job-{method}_{dataset}")
        tk.register_output(f"{method}-{dataset}-bart-base", train_job_lm.out_best_model)

        method = "noisy_channel_reranking"
        config["method"] = method
        config["per_device_eval_batch_size"] = 2

        cm_factor, lm_factor = 0.2, 0.5

        checkpoint = CreateNoisyChannelCheckpointJob(
            code_root,
            train_job_dm.out_best_model,
            train_job_cm.out_best_model,
            train_job_lm.out_best_model,
            cm_factor,
            lm_factor, # lm_factor
            1.0, # length penalty
        )

        search_job = HuggingfaceSearchJob(
            code_root=code_root,
            model_path=checkpoint.out_model_path,
            config=config,
            search_data_config=search_data_config,
            mem_rqmt=16,
            time_rqmt=8
        )

        tk.register_output(f'baselines/{method}_{dataset}_test.out.json', search_job.out_search_file)
        tk.register_output(f'baselines/{method}_{dataset}_test.metrics.json', search_job.out_metric_file)

        method = "online_channel_model"
        config["method"] = method

        train_job_cm = HuggingfaceTrainingJob(
            code_root=code_root,
            config=config,
            train_data_config=train_data_config,
            num_epochs=1,
            mem_rqmt=18,
            time_rqmt=24,
        )
        train_job_cm.add_alias(f"train_job-{method}_{dataset}")
        tk.register_output(f"{method}-{dataset}-bart-base", train_job_cm.out_best_model)

        method = "noisy_channel_online"
        config["method"] = method
        config["per_device_eval_batch_size"] = 2

        cm_factor, lm_factor = 0.5, 0.2

        checkpoint = CreateNoisyChannelCheckpointJob(
            code_root,
            train_job_dm.out_best_model,
            train_job_cm.out_best_model,
            train_job_lm.out_best_model,
            cm_factor,
            lm_factor, # lm_factor
            1.0, # length penalty
        )

        search_job = HuggingfaceSearchJob(
            code_root=code_root,
            model_path=checkpoint.out_model_path,
            config=config,
            search_data_config=search_data_config,
            mem_rqmt=16,
            time_rqmt=8
        )

        tk.register_output(f'baselines/{method}_{dataset}_test.out.json', search_job.out_search_file)
        tk.register_output(f'baselines/{method}_{dataset}_test.metrics.json', search_job.out_metric_file)

        method = "noisy_channel_liuetal"
        config["method"] = method
        config["generation_k_2"] = 2
        config['per_device_eval_batch_size'] = 1

        search_job = HuggingfaceSearchJob(
            code_root=code_root,
            model_path=checkpoint.out_model_path,
            config=config,
            search_data_config=search_data_config,
            mem_rqmt=16,
            time_rqmt=8
        )

        tk.register_output(f'baselines/{method}_{dataset}_test.out.json', search_job.out_search_file)
        tk.register_output(f'baselines/{method}_{dataset}_test.metrics.json', search_job.out_metric_file)

        del config["generation_k_2"]

        method = "ctrl"
        config["method"] = method
        config['per_device_eval_batch_size'] = 2

        for task_config in [train_data_config, search_data_config]:
            task_config["dataset_config_name"] = "control"

        train_job_dm = HuggingfaceTrainingJob(
            code_root=code_root,
            config=config,
            train_data_config=train_data_config,
            num_epochs=1,
            mem_rqmt=12,
            time_rqmt=24,
            gpu_rqmt=1
        )
        train_job_dm.add_alias(f"baselines/train_job-{method}_{dataset}")
        tk.register_output(f"baselines/{method}-{dataset}-bart-base", train_job_dm.out_best_model)

        search_job = HuggingfaceSearchJob(
            code_root=code_root,
            model_path=train_job_dm.out_best_model,
            config=config,
            search_data_config=search_data_config,
            mem_rqmt=8,
        )
        tk.register_output(f'baselines/{method}_{dataset}_test.out.json', search_job.out_search_file)
        tk.register_output(f'baselines/{method}_{dataset}_test.metrics.json', search_job.out_metric_file)

async def async_main():
    await baseline()

async def py():
    await async_main()
