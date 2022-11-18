import copy
import os
import shutil
import subprocess as sp
import copy
import json

from sisyphus import *

import i6_core.util as util
from i6_core.returnn.config import instanciate_delayed

Path = setup_path(__package__)


class CreateNoisyChannelCheckpointJob(Job):

  def __init__(
    self,
    code_root,
    dm_model_name_or_path,
    cm_model_name_or_path,
    lm_model_name_or_path,
    cm_scaling_factor,
    lm_scaling_factor,
    length_penalty,
    time_rqmt=4,
    mem_rqmt=18,
    cpu_rqmt=1,
    gpu_rqmt=1,
  ):
    self.code_root = code_root
    self.dm_model_name_or_path = dm_model_name_or_path
    self.cm_model_name_or_path = cm_model_name_or_path
    self.lm_model_name_or_path = lm_model_name_or_path
    self.cm_scaling_factor = cm_scaling_factor
    self.lm_scaling_factor = lm_scaling_factor
    self.length_penalty = length_penalty

    self.out_model_path = self.output_path("model")

    self.rqmt = {
      "gpu": gpu_rqmt,
      "cpu": cpu_rqmt,
      "mem": mem_rqmt,
      "time": time_rqmt,
    }

  def _get_run_cmd(self):
      args = [
          "--dm_model_name_or_path", str(self.dm_model_name_or_path),
          "--cm_model_name_or_path", str(self.cm_model_name_or_path),
          "--lm_model_name_or_path", str(self.lm_model_name_or_path),
          "--cm_scaling_factor", str(self.cm_scaling_factor),
          "--lm_scaling_factor", str(self.lm_scaling_factor),
          "--length_penalty", str(self.length_penalty),
          "--checkpoint_path", self.out_model_path.get_path()
      ]
      run_cmd = [
          tk.uncached_path(gs.PYTHON_EXE),
          str(os.path.join(tk.uncached_path(self.code_root), "i6_noisy_channel/models/create_nc_checkpoint.py")),
          *args
      ]
      return run_cmd

  def run(self):
      sp.check_call(self._get_run_cmd())

  def create_files(self):
    util.create_executable("run.sh", self._get_run_cmd())

  def tasks(self):
    yield Task("create_files", mini_task=True)
    yield Task("run", resume="run", rqmt=self.rqmt)

  @classmethod
  def hash(cls, kwargs):
      hash_kwargs = copy.deepcopy(kwargs)
      excluded_keys = ['time_rqmt', 'mem_rqmt', 'cpu_rqmt', 'gpu_rqmt']
      for key in excluded_keys:
        if key in hash_kwargs:
          del hash_kwargs[key]

      return super().hash(hash_kwargs)


class HuggingfaceTrainingJob(Job):
  """
  Train a Huggingface transformer model

  """
  __sis_hash_exclude__ = {
    'keep_only_best': False,
    'distributed': False,
    'qsub_args': None,
  }

  def __init__(
      self,
      code_root,
      config,
      train_data_config,
      *,  # args below are keyword only
      num_epochs=1,
      eval_metric="eval_loss",
      eval_metric_select_smallest=True,
      time_rqmt=4,
      mem_rqmt=4,
      cpu_rqmt=2,
      gpu_rqmt=1,
      qsub_args=None,
      python_exe=None,
      keep_only_best=False,
      distributed=False,
      **kwargs,
  ):
    """

    :param code_root: Root directory for the training scripts. Expected to contain a training script.
    :param config:
    :param num_epochs:
    :param time_rqmt:
    :param mem_rqmt:
    :param cpu_rqmt:
    :param gpu_rqmt:
    """

    self.code_root = code_root
    if "evaluation_strategy" not in config:
        config["evaluation_strategy"] = "epoch"
    self.config = config
    self.train_data_config = train_data_config
    self.python_exe = (python_exe if python_exe is not None else gs.PYTHON_EXE)

    self.num_epochs = num_epochs
    self.eval_metric = eval_metric
    self.eval_metric_select_smallest = eval_metric_select_smallest

    self.keep_only_best = keep_only_best
    self.distributed = distributed

    if gpu_rqmt > 1:
      qsub_args = "-P multigpu"
    elif qsub_args is None:
      qsub_args = []

    self.rqmt = {
      "gpu": gpu_rqmt,
      "cpu": cpu_rqmt,
      "mem": mem_rqmt,
      "time": time_rqmt,
      "qsub_args": qsub_args,
    }

    self.out_config_file = self.output_path("train_config.json")
    self.out_checkpoints_dir = self.output_path("checkpoints", directory=True)

    self.out_trainer_state = self.output_path("checkpoints/trainer_state.json")
    self.out_train_results = self.output_path("checkpoints/train_results.txt")

    # TODO do we want to handle this differently?
    self.out_models_dir = self.output_path("models", directory=True)
    if not keep_only_best:
      self.out_models = {
        k: self.output_path(f"models/epoch-{k:03}")
        for k in range(1, num_epochs + 1)
      }
    self.out_best_model = self.output_path("models/epoch-best")

    self._update_config()

  def _update_config(self):
    fixed_config = {
      'output_dir': self.out_checkpoints_dir.get_path(),
      'num_train_epochs': self.num_epochs,
      'logging_strategy': 'steps',
      'logging_steps': 128,
      'save_strategy': 'epoch',
      'overwrite_output_dir': True,
    }
    assert fixed_config.keys().isdisjoint(self.config.keys())
    self.config.update(fixed_config)
    assert self.config.keys().isdisjoint(self.train_data_config.keys())

  def _get_run_cmd(self):
      run_cmd = [
          tk.uncached_path(self.python_exe),
          os.path.join(tk.uncached_path(self.code_root), "train.py"),
          self.out_config_file.get_path(),
      ]
      if self.distributed:
        run_cmd = [
          tk.uncached_path(self.python_exe),
          '-m', 'torch.distributed.launch', '--nproc_per_node', str(self.rqmt['gpu']),
        ] + run_cmd[1:]
      return run_cmd

  def create_files(self):
    instanciated_config = instanciate_delayed({
      **copy.deepcopy(self.config),
      **copy.deepcopy(self.train_data_config),
    })
    with util.uopen(self.out_config_file, 'wt') as fp:
      json.dump(instanciated_config, fp)

    util.create_executable("run.sh", self._get_run_cmd())

  def run(self):
    sp.check_call(self._get_run_cmd())

  def cleanup(self):
    with util.uopen(self.out_trainer_state) as fp:
      trainer_state = json.load(fp)
      epoch_stats = trainer_state['log_history'][:-1]  # Remove the last global stat
      epoch_stats = list(filter(lambda x: self.eval_metric in x and x['epoch'].is_integer(), epoch_stats))
      assert len(epoch_stats) == self.num_epochs
      selection_method = max if not self.eval_metric_select_smallest else min
      best_epoch = selection_method(epoch_stats, key=lambda x: x[self.eval_metric])
      os.symlink(
        os.path.join(self.out_checkpoints_dir.get_path(), f"checkpoint-{best_epoch['step']}"),
        self.out_best_model.get_path(), target_is_directory=True)

      for epoch_stat in epoch_stats:
        if not self.keep_only_best:
          os.symlink(
            os.path.join(self.out_checkpoints_dir.get_path(), f"checkpoint-{epoch_stat['step']}"),
            self.out_models[int(epoch_stat['epoch'])].get_path(), target_is_directory=True)
        elif epoch_stat['step'] != best_epoch['step']:
          shutil.rmtree(os.path.join(self.out_checkpoints_dir.get_path(), f"checkpoint-{epoch_stat['step']}"))

  def tasks(self):
    yield Task("create_files", mini_task=True)
    yield Task("run", resume="run", rqmt=self.rqmt)
    yield Task("cleanup", mini_task=True)

  @classmethod
  def hash(cls, kwargs):
      hash_kwargs = copy.deepcopy(kwargs)
      excluded_keys = ['time_rqmt', 'mem_rqmt', 'cpu_rqmt', 'gpu_rqmt']
      for key in excluded_keys:
        if key in hash_kwargs:
          del hash_kwargs[key]

      if 'kwargs' in hash_kwargs and (hash_kwargs['kwargs'] is None or len(hash_kwargs['kwargs']) == 0):
        del hash_kwargs['kwargs']

      return super().hash(hash_kwargs)
