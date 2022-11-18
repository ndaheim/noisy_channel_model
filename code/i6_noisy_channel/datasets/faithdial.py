import itertools
import json
import os
from typing import List

import datasets
from datasets import load_dataset
from .base import DSTCBase

_URLs = {
    "train": "https://huggingface.co/datasets/McGill-NLP/FaithDial/resolve/main/data/train.json",
    "validation": "https://huggingface.co/datasets/McGill-NLP/FaithDial/resolve/main/data/valid.json",
    "test": "https://huggingface.co/datasets/McGill-NLP/FaithDial/resolve/main/data/test.json"
}

class FaithDial(DSTCBase, datasets.GeneratorBasedBuilder):

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
                    "control_tokens": datasets.Value("string"),
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
            if turn["speaker"] == "S":
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
                        # "score": 1.0
                    }
                    out.append(sample)

        return out


    def _map_to_dstc_format(self, dialog, prediction=None):
        out = []

        for i, turn_to_predict in enumerate(dialog["utterances"]):
            sample = { 
                "target": True,
                "knowledge": [
                {
                    "domain": None,
                    "entity_id": None, #if self.config.name == "selection" else None,
                    "doc_id": None,
                    "entity_name": None,
                    "title": None,
                    "body": turn_to_predict["knowledge"],
                    "score": 1.0
                }
                ],
                "response": turn_to_predict["response"],
                "turns": [
                    {
                        "speaker": "U" if i % 2 == 0 else "S",
                        "text": turn
                    }
                    for i, turn in enumerate(turn_to_predict["history"])
                ],
                "source": "faithdial"
            }
            
            if self.config.name == "control":
                if "control_tokens" in turn_to_predict:
                    control_tokens = " ".join(turn_to_predict["control_tokens"])
                else:
                    control_tokens = ""
                sample["control_tokens"] = control_tokens
                    
            out.append(sample)

        return out

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        # data = load_dataset("McGill-NLP/FaithDial")
                        
        splits = ["train", "validation", "test"]
        hf_splits = [datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST]
        data = {}
        
        for split in splits:
            url_to_download = _URLs[split]
            data_path = self._download_files(url_to_download, self.config.data_files, dl_manager)
            # data_path = data_path["data"]
        
            with open(data_path, "r") as f:
                print(data_path)
                data[split] = json.load(f)
                    
        return [
            datasets.SplitGenerator(
                name=ds_split, gen_kwargs={
                    "data": data[split],
                })
            for ds_split, split in zip(hf_splits, splits)
        ]

    def _generate_examples(self, data, predictions=None):
            
        if predictions is not None:
            data = list(itertools.chain.from_iterable([self._map_to_dstc_format(dialog, prediction=prediction) 
                                                        for dialog, prediction in zip(data, predictions)]))
        else:
            data = list(itertools.chain.from_iterable([self._map_to_dstc_format(dialog) for dialog in data]))

        for idx, sample in enumerate(data):
            if not "id" in sample:
                sample["id"] = str(idx)
            yield idx, sample
