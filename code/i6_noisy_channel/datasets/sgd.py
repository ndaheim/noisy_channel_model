import itertools
import json
import os
from typing import List

import datasets
from datasets import load_dataset, concatenate_datasets
from .base import DSTCBase

_BASE_URL = "https://raw.githubusercontent.com/google-research-datasets/Taskmaster/master/TM-1-2019/self-dialogs.json"

class SchemaGuidedDialog(DSTCBase, datasets.GeneratorBasedBuilder):

    def _info(self):

        if self.config.name in ["generation", "evaluation"]:
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
        else:
            raise NotImplementedError()

        return datasets.DatasetInfo(
            description="",
            features=features,
            supervised_keys=None,
            homepage="",
            citation="",
        )

    def _collapse_turns(self, turns):
        if len(turns) == 0:
            return turns

        out = []
        current_speaker = turns[0]["speaker"]
        current_turn = turns[0]

        for turn in turns[1:]:
            if turn["speaker"] == current_speaker:
                current_turn["text"] = current_turn["text"] + " " + turn["text"]
            else:
                out.append(current_turn)
                current_speaker = turn["speaker"]
                current_turn = turn
        out.append(current_turn)

        return out

    def _map_to_dstc_format(self, dialog):
        history = []
        out = []

        for i, turn in enumerate(dialog["turns"]["utterance"]):
            speaker = "S" if dialog["turns"]["speaker"][i] == 1 else "U"
            turn = {
                "speaker": speaker,
                "text": turn
            }
            history.append(turn)
        
        for i, turn in enumerate(history):
            if turn["speaker"] == "S":

                sample = { 
                    "target": True,
                    "knowledge": [
                    {
                        "domain": None,
                        "entity_id": None,
                        "doc_id": 0,
                        "entity_name": None,
                        "title": None,
                        "body": "",
                        "score": 1.0
                    }
                    ],
                    "response": turn["text"],
                    "turns": history[:i],
                    "source": "sgd"
                }
                out.append(sample)

        return out

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        data = {"train": [], "validation": [], "test": []}

        splits = ["train", "validation", "test"]
        dataset = load_dataset("schema_guided_dstc8")
        for split in splits:
            split_data = dataset[split]
            for sample in split_data:
                data[split].append(sample)

        return [
            datasets.SplitGenerator(
                name=ds_split, gen_kwargs={
                    "data": data[split],
                })
            for ds_split, split in zip([datasets.Split.TRAIN], splits)
        ]

    def _generate_examples(self, data):
        data = list(itertools.chain.from_iterable([self._map_to_dstc_format(dialog) for dialog in data]))

        for idx, sample in enumerate(data):
            sample["id"] = str(idx)
            yield idx, sample
