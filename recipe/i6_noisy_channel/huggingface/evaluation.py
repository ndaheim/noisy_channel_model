import copy
import json
import os
import subprocess as sp

from sisyphus import *

import i6_core.util as util
from i6_core.returnn.config import instanciate_delayed
from i6_core.tools.download import DownloadJob

Path = setup_path(__package__)


class CalculateDSTC10MetricsJob(Job):
    """
    Calculates all metrics for the tasks of detection, selection and generation 
    of the DSTC10 shared task.
    """

    def __init__(
        self,
        code_root,
        dataset_name,
        split,
        model_output_file,
        *,  # args below are keyword only
        dataset_filter_dict=None,
        time_rqmt=1,
        mem_rqmt=1,
        cpu_rqmt=1,
        gpu_rqmt=1,
        python_exe=None,
        **kwargs
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
        self.dataset = dataset_name
        self.dataset_filter_dict = dataset_filter_dict
        self.model_output_file = model_output_file
        self.split = split

        self.rqmt = {
            "gpu": gpu_rqmt,
            "cpu": cpu_rqmt,
            "mem": mem_rqmt,
            "time": time_rqmt,
        }
        self.out_results_file = self.output_path("metrics.json")
        self.out_dataset_filter_json = self.output_path("dataset_filter_dict.json")

        self.python_exe = os.path.join(*[self.code_root, "i6_noisy_channel", "score.py"])

    def _get_run_cmd(self):
        args = [
            "--dataset", self.dataset,
            "--split", self.split,
            "--outfile", self.model_output_file,
            "--scorefile", self.out_results_file,
        ]
        if self.dataset_filter_dict is not None:
            args.extend(["--dataset_filter_json", self.out_dataset_filter_json])
        run_cmd = [
            tk.uncached_path(gs.PYTHON_EXE),
            os.path.join(tk.uncached_path(self.code_root), "i6_noisy_channel/score.py"),
            *args
        ]
        return run_cmd

    def run(self):
        if self.dataset_filter_dict is not None:
            import json
            with open(self.out_dataset_filter_json, "w") as f:
                json.dump(self.dataset_filter_dict, f)

        sp.check_call(self._get_run_cmd())

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt, mini_task=False)

    @classmethod
    def hash(cls, kwargs):
        hash_kwargs = copy.deepcopy(kwargs)
        excluded_keys = ['time_rqmt', 'mem_rqmt', 'cpu_rqmt']
        for key in excluded_keys:
            if key in hash_kwargs:
                del hash_kwargs[key]

        return super().hash(hash_kwargs)

class CalculateQ2Job(Job):
    """
    """

    def __init__(
        self,
        code_root,
        dataset_name,
        split,
        model_output_file,
        *,  # args below are keyword only
        dataset_filter_dict=None,
        time_rqmt=3,
        mem_rqmt=12,
        cpu_rqmt=1,
        gpu_rqmt=1,
        python_exe=None,
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
        self.dataset = dataset_name
        self.dataset_filter_dict = dataset_filter_dict
        self.model_output_file = model_output_file
        self.split = split

        self.rqmt = {
            "gpu": gpu_rqmt,
            "cpu": cpu_rqmt,
            "mem": mem_rqmt,
            "time": time_rqmt,
        }
        self.out_q2_input_file = self.output_path("model_output.csv")
        self.out_q2_output_file = self.output_path("q2_output")
        self.out_results_file = self.output_path("metrics.csv")
        self.out_dataset_filter_json = self.output_path("dataset_filter_dict.json")

    def _get_run_cmd(self):
        args = [
            "--infile", self.out_q2_input_file.get_path(),
            "--gen_method", "beam",
            "--q_per_cand", "single",
            "--personal", "remove",
            "--outfile", self.out_q2_output_file.get_path(),
            "--save_steps"
        ]
        run_cmd = [
            tk.uncached_path(gs.PYTHON_EXE),
            os.path.join(tk.uncached_path(gs.Q2_PATH), "pipeline/run_pipeline.py"),
            *args
        ]
        return run_cmd

    def run(self):
        if self.dataset_filter_dict is not None:
            import json
            with open(self.out_dataset_filter_json, "w") as f:
                json.dump(self.dataset_filter_dict, f)

        sp.check_call(self._get_run_cmd())

    def calculate_scores(self):
        args = [
            "--infile", self.out_q2_output_file.get_path() + ".steps.csv",
            "--outfile", self.out_results_file.get_path(),
            "--task", "span_comparison"
        ]
        run_cmd = [
            tk.uncached_path(gs.PYTHON_EXE),
            os.path.join(tk.uncached_path(gs.Q2_PATH), "run_nli.py"),
            *args
        ]
        sp.check_call(run_cmd)

    def create_files(self):
        from datasets import load_dataset
        import pandas as pd
        # dataset = load_dataset(self.dataset, "evaluation", split=self.split)
        dataset = load_dataset(self.dataset, "evaluation", split=self.split)
        out = {"response": [], "knowledge": [], "gold": []}
        with open(self.model_output_file, "r") as f:
            model_output = json.load(f)

        for output_sample, input_sample in zip(model_output, dataset):
            if "target" in input_sample and "target" in output_sample:
                if output_sample["target"] and input_sample["target"]:
                    out["response"].append(output_sample["response"])
                    out["knowledge"].append(input_sample["knowledge"][0]["body"])
                    out["gold"].append(input_sample["response"])

        pd.DataFrame.from_dict(out).to_csv(self.out_q2_input_file)
        util.create_executable("run.sh", self._get_run_cmd())

    def tasks(self):
        yield Task("create_files", rqmt=self.update_rqmt_pure(time_rqmt=1))
        yield Task("run", resume="run", rqmt=self.rqmt, mini_task=False)
        yield Task("calculate_scores", rqmt=self.rqmt)

    def update_rqmt_pure(self, **kwargs):
        rqmt = self.rqmt.copy()
        
        for key in rqmt.keys():
            value = kwargs.get(key, None)
            if value is not None:
                rqmt[key] = value

        return rqmt

    @classmethod
    def hash(cls, kwargs):
        hash_kwargs = copy.deepcopy(kwargs)
        excluded_keys = ['time_rqmt', 'mem_rqmt', 'cpu_rqmt']
        for key in excluded_keys:
            if key in hash_kwargs:
                del hash_kwargs[key]

        return super().hash(hash_kwargs)


class ConvertPredictionsToGemFormatJob(Job):

    def __init__(self, predictions):
        self.predictions = predictions
        self.out_file = self.output_path("predictions.json")
        self.rqmt = {
            "gpu": 0,
            "cpu": 1,
            "mem": 4,
            "time": 1,
        }

    def run(self):
        out = {
            "values": [],
            "language": "en"
        }
        with open(self.predictions, "r") as f:
            predictions = json.load(f)

        for prediction in predictions:
            if prediction["target"]:
                out["values"].append(prediction["response"])
                
        with open(self.out_file, "w") as f:
            json.dump(out, f)

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt, mini_task=True)

    @classmethod
    def hash(cls, kwargs):
        hash_kwargs = copy.deepcopy(kwargs)
        excluded_keys = ['time_rqmt', 'mem_rqmt', 'cpu_rqmt', 'gpu_rqmt']
        for key in excluded_keys:
            if key in hash_kwargs:
                del hash_kwargs[key]

        return super().hash(hash_kwargs)

class ConvertLabelsToGemFormatJob(Job):

    def __init__(
        self, 
        dataset_name, 
        dataset_config_name, 
        split, 
        dataset_data_files=None, 
        dataset_filter_dict=None
    ):
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.split=split
        self.dataset_data_files = dataset_data_files
        self.dataset_filter_dict = dataset_filter_dict
        self.out_file = self.output_path("labels.json")
        self.rqmt = {
            "gpu": 0,
            "cpu": 1,
            "mem": 4,
            "time": 1,
        }

    def run(self):
        import json
        out = {
            "values": [],
            "language": "en"
        }
        from datasets import load_dataset
        dataset: Optional[Dataset] = load_dataset(
            self.dataset_name,
            self.dataset_config_name,
            split=self.split,
            data_files=self.dataset_data_files,
            dataset_filter_dict=self.dataset_filter_dict
        )

        for sample in dataset:
            if sample["target"]:
                out["values"].append({
                    "target": [sample["response"]]
                })

        with open(self.out_file, "w") as f:
            json.dump(out, f)

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt, mini_task=True)

    @classmethod
    def hash(cls, kwargs):
        hash_kwargs = copy.deepcopy(kwargs)
        excluded_keys = ['time_rqmt', 'mem_rqmt', 'cpu_rqmt', 'gpu_rqmt']
        for key in excluded_keys:
            if key in hash_kwargs:
                del hash_kwargs[key]

        return super().hash(hash_kwargs)

class CalculateGemMetricsJob(Job):

    def __init__(self, gem_path, labels, predictions, heavy_metrics=False):
        self.gem_path = gem_path
        self.labels = labels
        self.predictions = predictions
        self.heavy_metrics = heavy_metrics
        self.out_file = self.output_path("metrics.json")
        self.rqmt = {
            "gpu": int(self.heavy_metrics),
            "cpu": 1,
            "mem": 16 if self.heavy_metrics else 4,
            "time": 4,
        }

    def _get_run_cmd(self):
        run_cmd = [
            tk.uncached_path(gs.PYTHON_EXE),
            f"{gem_path}/run_metrics.py",
            "-r",
            self.labels.get_path(),
            "-o",
            self.out_file.get_path(),
            self.predictions.get_path()
        ]
        if self.heavy_metrics:
            run_cmd = run_cmd[:2] + ["--heavy-metrics"] + run_cmd[2:]
        return run_cmd

    def run(self):
        sp.check_call(self._get_run_cmd())

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt, mini_task=not self.heavy_metrics)

    @classmethod
    def hash(cls, kwargs):
        hash_kwargs = copy.deepcopy(kwargs)
        excluded_keys = ['time_rqmt', 'mem_rqmt', 'cpu_rqmt', 'gpu_rqmt']
        for key in excluded_keys:
            if key in hash_kwargs:
                del hash_kwargs[key]

        return super().hash(hash_kwargs)

class CalculateNubiaJob(Job):

    def __init__(self, labels, predictions):
        self.labels = labels
        self.predictions = predictions
        self.out_file = self.output_path("metrics.json")
        self.rqmt = {
            "gpu": 0,
            "cpu": 1,
            "mem": 16,
            "time": 12,
        }

    def run(self):
        out = {
            'nubia_score': 0.0,
            'features': {
                'semantic_relation': 0.0,
                'contradiction': 0.0,
                'irrelevancy': 0.0,
                'logical_agreement': 0.0,
                'grammar_ref': 0.0,
                'grammar_hyp': 0.0
                }
            }
        from nubia_score import Nubia
        from tqdm import tqdm
        import json
        nubia = Nubia()

        with open(self.labels) as f:
            labels = json.load(f)["values"]

        with open(self.predictions) as f:
            predictions = json.load(f)["values"]

        for label, pred in tqdm(zip(labels, predictions)):
            score = nubia.score(label["target"][0], pred, get_features=True)
            out["nubia_score"] += score["nubia_score"]
            for key in out["features"].keys():
                out["features"][key] += score["features"][key]

        out["nubia_score"] = out["nubia_score"] / len(labels)
        for key in out["features"].keys():
            out["features"][key] = out["features"][key] / len(labels)

        with open(self.out_file, "w") as f:
            json.dump(out, f)

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt, mini_task=False)

    @classmethod
    def hash(cls, kwargs):
        hash_kwargs = copy.deepcopy(kwargs)
        excluded_keys = ['time_rqmt', 'mem_rqmt', 'cpu_rqmt', 'gpu_rqmt']
        for key in excluded_keys:
            if key in hash_kwargs:
                del hash_kwargs[key]

        return super().hash(hash_kwargs)

class MergePredictionsJob(Job):
    __sis_hash_exclude__ = {
        'dataset_filter_dict': None,
    }

    def __init__(
        self, 
        files,
        dataset_name,
        split,
        dataset_config_name="detection",
        data_files=None,
        target_key="knowledge",
        *,  # args below are keyword only
        dataset_filter_dict=None,
        time_rqmt=1,
        mem_rqmt=1,
        cpu_rqmt=1,
        gpu_rqmt=0,
    ):
        self.files = files
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.split = split
        self.data_files = data_files
        self.target_key = target_key
        self.dataset_filter_dict = dataset_filter_dict

        self.rqmt = {
            "gpu": gpu_rqmt,
            "cpu": cpu_rqmt,
            "mem": mem_rqmt,
            "time": time_rqmt,
        }
        self.out_predictions_file = self.output_path("predictions.json")

    def run(self):
        import itertools
        import json

        from datasets import load_dataset

        full_dataset = load_dataset(
            self.dataset_name,
            self.dataset_config_name,
            split=self.split,
            data_files=instanciate_delayed(self.data_files),
            **(
                {'dataset_filter_dict': self.dataset_filter_dict}
                if self.dataset_filter_dict is not None else {}
            ),
        )

        predictions = []
        for file in self.files:
            with open(file, "r") as f:
                predictions.append(json.load(f))

        selection_results = list(itertools.chain(*predictions))

        out = []
        idx = 0

        for sample in full_dataset:
            item = {}
            item["target"] = sample["target"]

            if idx < len(selection_results) and item["target"]:
                item["knowledge"] = selection_results[idx]
                idx += 1

            out.append(item)    

        with open(self.out_predictions_file, "w") as f:
            json.dump(out, f)

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt, mini_task=True)

    @classmethod
    def hash(cls, kwargs):
        hash_kwargs = copy.deepcopy(kwargs)
        excluded_keys = ['time_rqmt', 'mem_rqmt', 'cpu_rqmt', 'gpu_rqmt']
        for key in excluded_keys:
            if key in hash_kwargs:
                del hash_kwargs[key]

        return super().hash(hash_kwargs)


class MergeDSTC9TestSplitsJob(Job):
    def __init__(
        self,
        mwoz_predictions_file,
        sf_written_predictions_file,
        sf_spoken_predictions_file,
        dstc9_test_label_file=None,
    ):
        self.mwoz_predictions_file = mwoz_predictions_file
        self.sf_written_predictions_file = sf_written_predictions_file
        self.sf_spoken_predictions_file = sf_spoken_predictions_file
        self.dstc9_test_label_file = dstc9_test_label_file if dstc9_test_label_file is not None \
            else DownloadJob("https://github.com/alexa/alexa-with-dstc9-track1-dataset/raw/master/data_eval/test"
                             "/labels.json").out_file

        self.out_predictions_file = self.output_path('predictions.json')

    def run(self):
        with util.uopen(self.mwoz_predictions_file) as fp:
            mwoz_predictions = json.load(fp)
        with util.uopen(self.sf_written_predictions_file) as fp:
            sf_written_predictions = json.load(fp)
        with util.uopen(self.sf_spoken_predictions_file) as fp:
            sf_spoken_predictions = json.load(fp)

        with util.uopen(self.dstc9_test_label_file) as fp:
            test_label = json.load(fp)

        output_labels = []

        predictions_iters = {
            'multiwoz': iter(mwoz_predictions),
            'sf_written': iter(sf_written_predictions),
            'sf_spoken': iter(sf_spoken_predictions),
        }

        for label in test_label:
            output_labels.append(
                next(predictions_iters[label['source']])
            )

        # Check that all inputs are consumed
        for pred_iter in predictions_iters.values():
            assert not any(True for _ in pred_iter)

        with util.uopen(self.out_predictions_file, 'w') as fp:
            json.dump(output_labels, fp)

    def tasks(self):
        yield Task('run', mini_task=True)


class CalculatePerplexityJob(Job):
  """
  """

  def __init__(
      self,
      code_root,
      model_path,
      config,
      search_data_config,
      *,  # args below are keyword only
      time_rqmt=4,
      mem_rqmt=4,
      cpu_rqmt=2,
      gpu_rqmt=1,
      python_exe=None,
      **kwargs
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
    self.model_path = model_path
    self.config = config
    self.search_data_config = search_data_config
    self.python_exe = (python_exe if python_exe is not None else gs.PYTHON_EXE)

    self.rqmt = {
      "gpu": gpu_rqmt,
      "cpu": cpu_rqmt,
      "mem": mem_rqmt,
      "time": time_rqmt,
    }

    self.out_config_file = self.output_path("search_config.json")
    self.out_search_file = self.output_path("perplexity.json")

    self._update_config()

  def _update_config(self):
    fixed_config = {
      'prediction_output_file': self.out_search_file,
      'output_dir': 'trainer_output_dir',
    }
    assert fixed_config.keys().isdisjoint(self.config.keys())
    self.config = copy.deepcopy(self.config)
    self.config.update(fixed_config)
    # Overwrite model path
    self.config['model_name_or_path'] = self.model_path
    self.config['config_name'] = None
    self.config['tokenizer_name'] = None
    assert self.config.keys().isdisjoint(self.search_data_config.keys())

  def _get_run_cmd(self):
      run_cmd = [
          tk.uncached_path(self.python_exe),
          os.path.join(tk.uncached_path(self.code_root), "perplexity.py"),
          self.out_config_file.get_path(),
      ]
      return run_cmd

  def create_files(self):
    instanciated_config = instanciate_delayed({
      **copy.deepcopy(self.config),
      **copy.deepcopy(self.search_data_config),
    })
    with util.uopen(self.out_config_file, 'wt') as fp:
      json.dump(instanciated_config, fp)

    util.create_executable("run.sh", self._get_run_cmd())

  def run(self):
    sp.check_call(self._get_run_cmd())

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


class CalculateSearchErrorsJob(Job):
  """
  """

  def __init__(
      self,
      code_root,
      model_path,
      config,
      search_data_config,
      *,  # args below are keyword only
      time_rqmt=4,
      mem_rqmt=4,
      cpu_rqmt=2,
      gpu_rqmt=1,
      python_exe=None,
      **kwargs
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
    self.model_path = model_path
    self.config = config
    self.search_data_config = search_data_config
    self.python_exe = (python_exe if python_exe is not None else gs.PYTHON_EXE)

    self.rqmt = {
      "gpu": gpu_rqmt,
      "cpu": cpu_rqmt,
      "mem": mem_rqmt,
      "time": time_rqmt,
    }

    self.out_config_file = self.output_path("search_config.json")
    self.out_search_file = self.output_path("search_errors.json")

    self._update_config()

  def _update_config(self):
    fixed_config = {
      'prediction_output_file': self.out_search_file,
      'output_dir': 'trainer_output_dir',
    }
    assert fixed_config.keys().isdisjoint(self.config.keys())
    self.config = copy.deepcopy(self.config)
    self.config.update(fixed_config)
    # Overwrite model path
    self.config['model_name_or_path'] = self.model_path
    self.config['config_name'] = None
    self.config['tokenizer_name'] = None
    assert self.config.keys().isdisjoint(self.search_data_config.keys())

  def _get_run_cmd(self):
      run_cmd = [
          tk.uncached_path(self.python_exe),
          os.path.join(tk.uncached_path(self.code_root), "search_errors.py"),
          self.out_config_file.get_path(),
      ]
      return run_cmd

  def create_files(self):
    instanciated_config = instanciate_delayed({
      **copy.deepcopy(self.config),
      **copy.deepcopy(self.search_data_config),
    })
    with util.uopen(self.out_config_file, 'wt') as fp:
      json.dump(instanciated_config, fp)

    util.create_executable("run.sh", self._get_run_cmd())

  def run(self):
    sp.check_call(self._get_run_cmd())

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
