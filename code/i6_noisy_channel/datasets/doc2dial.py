import itertools
import json
import os
from typing import List

import datasets
from datasets import load_dataset
from .base import DSTCBase
from .ctrl_mixin import CTRLMixin

_URLs = {
    "train": "https://raw.githubusercontent.com/doc2dial/sharedtask-dialdoc2021/master/data/doc2dial/v1.0.1/doc2dial_dial_train.json",
    "validation": "https://raw.githubusercontent.com/doc2dial/sharedtask-dialdoc2021/master/data/doc2dial/v1.0.1/doc2dial_dial_validation.json",
    "test": "https://doc2dial.github.io/file/doc2dial_v1.0.1.zip",
    "docs": {
        "train": "https://raw.githubusercontent.com/doc2dial/sharedtask-dialdoc2021/master/data/doc2dial/v1.0.1/doc2dial_doc.json",
        "validation": "https://raw.githubusercontent.com/doc2dial/sharedtask-dialdoc2021/master/data/doc2dial/v1.0.1/doc2dial_doc.json",
        "test": "https://raw.githubusercontent.com/ndaheim/dialdoc-sharedtask-21/main/code/data/doc2dial/v1.0.1/test/doc2dial_doc_with_unseen.json"
    }
}

class Doc2Dial(DSTCBase, datasets.GeneratorBasedBuilder, CTRLMixin):

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

        for i, turn in enumerate(dialog["turns"]):
            speaker = "U" if turn["role"] == "user" else "S"
            references = turn["references"]
            turn = {
                "speaker": speaker,
                "text": turn["utterance"],
            }
            history.append(turn)
            if i > 0 and turn["speaker"] == "S":
                source = history[:i]

                text = dialog["doc"]["doc_text"]
                text = " ".join(set([dialog["doc"]["spans"][ref["sp_id"]]["text_sp"] for ref in references]))

                sample = { 
                    "target": True,
                    "knowledge": [
                    {
                        "domain": dialog["domain"],
                        "entity_id": None,
                        "doc_id": dialog["doc"]["numerical_id"],
                        "entity_name": None,
                        "title": dialog["doc"]["doc_id"],
                        "body": text,
                        "score": 1.0
                    }
                    ],
                    "response": turn["text"],
                    "turns": source,
                    "source": "doc2dial"
                }
                out.append(sample)

        return out

    def _download_and_extract_file(self, url, dl_manager):
        file_path = self._download_files(url, self.config.data_files, dl_manager)
        with open(file_path, "r") as f:
            return json.load(f)

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        idx = 0
        splits = ["train", "validation", "test"]
        data = {split: [] for split in splits}
        docs = {split: self._download_and_extract_file(_URLs["docs"][split], dl_manager)["doc_data"] for split in splits}

        for split in splits:
            url_to_download = _URLs[split]
            if split == "test":
                split_path = self._download_files(url_to_download, self.config.data_files, dl_manager)
                file_path = os.path.join(split_path, "doc2dial_dial_test.json")
                with open(file_path, "r") as f:
                    split_data = json.load(f)
            else:
                split_data = self._download_and_extract_file(url_to_download, dl_manager)

            for domain in split_data["dial_data"]:
                for doc_id in split_data["dial_data"][domain]:
                    for dialogue in split_data["dial_data"][domain][doc_id]:
                        doc = docs[split][domain][doc_id]
                        doc["numerical_id"] = idx
                        sample = {
                            "domain": domain,
                            "doc": doc,
                            "turns": dialogue["turns"],
                        }
                        data[split].append(sample)
                    idx += 1

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
