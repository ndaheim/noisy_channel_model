import itertools
import json
import os
from typing import List

import datasets
from datasets import load_dataset
from .base import DSTCBase
from .ctrl_mixin import CTRLMixin

_URL = "http://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz"

class WizardOfWikipedia(DSTCBase, datasets.GeneratorBasedBuilder, CTRLMixin):

    def _info(self):

        if self.config.name in ["generation", "evaluation", "selection"]:
            features = datasets.Features(
                {
                    "id": datasets.Value('string'),
                    "target": datasets.Value("bool"),
                    "knowledge": [
                        {
                            "domain": datasets.Value("string"),
                            "entity_id": datasets.Value("string"),
                            "doc_id": datasets.Value("int32"),
                            "entity_name": datasets.Value("string"),
                            "title": datasets.Value("string"),
                            "body": datasets.Value("string"),
                            "score": datasets.Value("float"),
                        }
                    ],
                    "response": datasets.Value("string"),
                    "turns": [
                        {
                            "speaker": datasets.Value("string"),
                            "text": datasets.Value("string"),
                        }
                    ],
                    "source": datasets.Value('string'),
                }
            )
        elif self.config.name == "control":
            features = datasets.Features(
                {
                    "id": datasets.Value('string'),
                    "target": datasets.Value("bool"),
                    "knowledge": [
                        {
                            "domain": datasets.Value("string"),
                            "entity_id": datasets.Value("string"),
                            "doc_id": datasets.Value("int32"),
                            "entity_name": datasets.Value("string"),
                            "title": datasets.Value("string"),
                            "body": datasets.Value("string"),
                            "score": datasets.Value("float"),
                        }
                    ],
                    "response": datasets.Value("string"),
                    "control_tokens": datasets.Value("string"),
                    "turns": [
                        {
                            "speaker": datasets.Value("string"),
                            "text": datasets.Value("string"),
                        }
                    ],
                    "source": datasets.Value('string'),
                }
            )
        elif self.config.name in ["knowledge", "knowledge_entities", "knowledge_domains"]:
            features = datasets.Features(
                {
                    "id": datasets.Value('string'),
                    "domain": datasets.Value("string"),
                    **({
                        "entity_id": datasets.Value("string"),
                        "entity_name": datasets.Value("string"),
                    } if self.config.name in ["knowledge", "knowledge_entities"] else {}),
                    **({
                        "doc_id": datasets.Value("int32"),
                        "title": datasets.Value("string"),
                        "body": datasets.Value("string"),
                    } if self.config.name == "knowledge" else {}),
                }
            )
        else:
            raise NotImplementedError()

        return datasets.DatasetInfo(
            description="",
            features=features,
            supervised_keys=None,
            homepage="",
            citation="",
        )

    def _get_knowledge(self, dialog):
        out = []

        for i, turn in enumerate(dialog["dialog"]):
            turn["speaker"] = "S" if turn["speaker"] == "0_Wizard" else "U"
            if i > 0 and turn["speaker"] == "S":
                if turn["checked_sentence"] != {} and turn["checked_sentence"] != {'no_passages_used': 'no_passages_used'}:
                    knowledge_items = list(turn["checked_sentence"].keys())[0].split("_")
                    domain, entity_name, doc_id = dialog["chosen_topic"], " ".join(knowledge_items[1:-1]), knowledge_items[-1]
                    sample = {
                        "id": f"{domain}__{entity_name}__{doc_id}",
                        "domain": domain,
                        "entity_id": entity_name,
                        "doc_id": int(doc_id),
                        "entity_name": entity_name,
                        "title": "",
                        "body": list(turn["checked_sentence"].values())[0],
                    }
                    out.append(sample)

        return out


    def _map_to_dstc_format(self, dialog, prediction=None):
        out = []

        for i, turn in enumerate(dialog["dialog"]):
            turn["speaker"] = "S" if turn["speaker"] == "0_Wizard" else "U"
            if turn["speaker"] == "S":
                if turn["checked_sentence"] != {} and turn["checked_sentence"] != {'no_passages_used': 'no_passages_used'}:
                    source = [{
                        "text": turn["text"].strip(),
                        "speaker": turn["speaker"]
                    } for turn in dialog["dialog"][:i]]

                    knowledge_items = list(turn["checked_sentence"].keys())[0].split("_")
                    if prediction is None:
                        domain, entity_name, doc_id = dialog["chosen_topic"], " ".join(knowledge_items[1:-1]), knowledge_items[-1]
                    else:
                        domain, entity_name, doc_id = prediction["knowledge"][0]["domain"], prediction["knowledge"][0]["entity_id"], prediction["knowledge"][0]["doc_id"]
                    sample = { 
                        "target": True,
                        "knowledge": [
                        {
                            "domain": domain,
                            "entity_id": entity_name,
                            "doc_id": int(doc_id),
                            "entity_name": entity_name,
                            "title": "" if self.config.name == "selection" else None,
                            "body": list(turn["checked_sentence"].values())[0] if prediction is None else prediction["knowledge"][0]["body"],
                            "score": 1.0
                        }
                        ],
                        "response": turn["text"].strip(),
                        "turns": source,
                        "source": "wow"
                    }
                    
                    out.append(sample)

        return out

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        url_to_download = _URL
        data_path = self._download_files(url_to_download, self.config.data_files, dl_manager)
        if isinstance(data_path, dict):
            test_predictions = data_path["test"]["predictions"]
            data_path = data_path["data"]
        else:
            test_predictions = {"train": None, "val": None, "test": None}
            
        splits = ["train", "val", "test"]

        file_names = ["train.json", "test_topic_split.json", "test_random_split.json"]
        
        data = {}
        file_paths = []
        for split, file_name in zip(splits, file_names):
            file_path = os.path.join(data_path, file_name)
            file_paths.append(file_path)
            with open(file_path, "r") as f:
                data[split] = json.load(f)
                
        if test_predictions != {"train": None, "val": None, "test": None}:
            with open(test_predictions, "r") as f:
                test_predictions = {"train": [], "val": [], "test": json.load(f)}
                
        if self.config.name in ["knowledge", "knowledge_entities", "knowledge_domains"]:
            new_data = {}
            new_data["train"] = data["train"]
            new_data["train"].extend(data["test"])
            new_data["val"] = data["val"]
            with open(file_paths[0]) as f:
                new_data["test"] = json.load(f)
            with open(file_paths[-1]) as f:
                new_data["test"].extend(json.load(f))
        else:
            new_data = data
                        
        hf_splits = [datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST]
        return [
            datasets.SplitGenerator(
                name=ds_split, gen_kwargs={
                    "data": new_data[split],
                    "predictions": test_predictions[split],
                })
            for ds_split, split in zip(hf_splits, splits)
        ]

    def _generate_examples(self, data, predictions=None):
            
        if self.config.name in ["knowledge", "knowledge_entities", "knowledge_domains"]:
            knowledge_data = []
            for dialog in data:
                knowledge = self._get_knowledge(dialog)
                for item in knowledge:
                    if not item in knowledge_data:
                        knowledge_data.append(item)
            data = knowledge_data
        else:
            if predictions is not None:
                data = list(itertools.chain.from_iterable([self._map_to_dstc_format(dialog, prediction=prediction) 
                                                           for dialog, prediction in zip(data, predictions)]))
            else:
                data = list(itertools.chain.from_iterable([self._map_to_dstc_format(dialog) for dialog in data]))

        if self.config.name == "control":
            data = self._add_control_tokens(data)

        for idx, sample in enumerate(data):
            if not "id" in sample:
                sample["id"] = str(idx)
            yield idx, sample
