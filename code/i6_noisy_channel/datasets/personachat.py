import itertools
import json
import os
from typing import List

import datasets
from datasets import load_dataset
from .base import DSTCBase
from .ctrl_mixin import CTRLMixin

_URL = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"

class PersonaChat(DSTCBase, datasets.GeneratorBasedBuilder, CTRLMixin):

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
        else:
            raise NotImplementedError()

        return datasets.DatasetInfo(
            description="",
            features=features,
            supervised_keys=None,
            homepage="",
            citation="",
        )

    def _map_to_dstc_format(self, dialog):
        history = []
        out = []

        for i, turn in enumerate(dialog["utterances"][-1]["history"]):
            speaker = "U" if i % 2 == 0 else "S"
            turn = {
                "speaker": speaker,
                "text": turn
            }
            history.append(turn)
            if turn["speaker"] == "S":
                source = history[:i]

                sample = { 
                    "target": True,
                    "knowledge": [
                    {
                        "domain": None,
                        "entity_id": None,
                        "doc_id": 0,
                        "entity_name": None,
                        "title": None,
                        "body": " ".join(dialog["personality"]),
                        "score": 1.0
                    }
                    ],
                    "response": turn["text"],
                    "turns": source,
                    "source": "personachat"
                }
                out.append(sample)

        return out

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        url_to_download = _URL
        file_path = self._download_files(url_to_download, self.config.data_files, dl_manager)
        with open(file_path, "r") as f:
            data = json.load(f)

        splits = ["train", "valid", "valid"]

        return [
            datasets.SplitGenerator(
                name=ds_split, gen_kwargs={
                    "data": data[split],
                })
            for ds_split, split in zip([datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST], splits)
        ]

    def _generate_examples(self, data):
        data = list(itertools.chain.from_iterable([self._map_to_dstc_format(dialog) for dialog in data]))
        if self.config.name == "control":
            data = self._add_control_tokens(data)

        for idx, sample in enumerate(data):
            sample["id"] = str(idx)                
            yield idx, sample
