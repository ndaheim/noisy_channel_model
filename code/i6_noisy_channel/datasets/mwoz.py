import itertools
import json
import os
from typing import List

import datasets
from datasets import load_dataset
from .base import DSTCBase

_URL = "https://github.com/budzianowski/multiwoz/raw/master/data/MultiWOZ_2.1.zip"

class MultiWoZ(DSTCBase, datasets.GeneratorBasedBuilder):

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

    def _get_dstc_mwoz_ids(self, data):
        dstc_val = datasets.load_dataset(
            None, # TODO: add path to dstc9_track1 dataset
            self.config.name, 
            split="validation"
        )
        dstc_test = datasets.load_dataset(
            None, # TODO: add path to dstc9_track1 dataset
            self.config.name, 
            split="test"
        )

        dstc_dataset = dstc_test
        real_knowledge_seeking_turns = {
            dstc_dial['turns'][-1]['text']
            for dstc_dial in dstc_dataset if dstc_dial['target']
        }
        knowledge_seeking_turns = real_knowledge_seeking_turns.copy()

        def map_func(dstc_dial):
            mwoz_id = None
            longest_match = 0
            longest_match_num_snippets = 0

            for mwoz_key, mwoz_dial in data.items():
                mwoz_turns = iter(mwoz_dial['log'])
                dstc_turns = iter(dstc_dial['turns'])
                match_length = 0
                num_snippets = 0

                try:
                    curr_mwoz_turn = next(mwoz_turns)
                    curr_dstc_turn = next(dstc_turns)
                    while True:
                        if curr_mwoz_turn['text'].strip() != curr_dstc_turn['text'].strip():
                            if curr_dstc_turn['text'] in knowledge_seeking_turns:
                                next(dstc_turns)
                                curr_dstc_turn = next(dstc_turns)
                                num_snippets += 1
                                continue
                            break
                        match_length += 1
                        curr_mwoz_turn = next(mwoz_turns)
                        curr_dstc_turn = next(dstc_turns)
                except Exception:
                # Reached end of dstc turns
                    mwoz_id = mwoz_key
                    break
                if longest_match < match_length:
                    longest_match = match_length
                    longest_match_num_snippets = num_snippets
                
            return {'mwoz_id': mwoz_id}

        dstc_dataset = dstc_dataset.map(map_func, batched=False, num_proc=1)
        return [dialog["mwoz_id"] for dialog in dstc_dataset]


    def _map_to_dstc_format(self, dialog):
        history = []
        out = []

        for i, turn in enumerate(dialog["log"]):
            if turn["metadata"] == {}:
                speaker = "U"
            else:
                speaker = "S"

            history.append({
                "text": turn["text"],
                "speaker": speaker
            })

            if speaker == "S":
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
                        "body": "",
                        "score": 1.0
                    }
                    ],
                    "response": turn["text"],
                    "turns": source,
                    "source": "mwoz"
                }
                out.append(sample)

        return out

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        url_to_download = _URL
        base_path = self._download_files(url_to_download, self.config.data_files, dl_manager)
        file_path = os.path.join(base_path, "MultiWOZ_2.1", "data.json")
        with open(file_path, "r") as f:
            data = json.load(f)
        
        file_path = os.path.join(base_path, "MultiWOZ_2.1", "valListFile.txt")
        with open(file_path, "r") as f:
            val_files = f.readlines()
            val_files = [file.strip() for file in val_files]
            
        file_path = os.path.join(base_path, "MultiWOZ_2.1", "testListFile.txt")
        with open(file_path, "r") as f:
            test_files = f.readlines()
            test_files = [file.strip() for file in test_files]
            
        splits = ["train", "val", "test"]
        new_data = {split: {} for split in splits}
        
        for file, sample in data.items():
            if file in val_files:
                new_data["val"][file] = sample
            elif file in test_files:
                new_data["test"][file] = sample
            else:
                new_data["train"][file] = sample

        return [
            datasets.SplitGenerator(
                name=ds_split, gen_kwargs={
                    "data": new_data[split],
                })
            for ds_split, split in zip([datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST], splits)
        ]

    def _generate_examples(self, data):
        ids_to_remove = self._get_dstc_mwoz_ids(data)

        data = [self._map_to_dstc_format(dialog) for key, dialog in data.items()
                if key not in ids_to_remove]
        data = list(itertools.chain.from_iterable(data))

        for idx, sample in enumerate(data):
            sample["id"] = str(idx)
            yield idx, sample
