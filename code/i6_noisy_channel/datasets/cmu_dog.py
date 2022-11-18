import itertools
import json
import os
from collections import defaultdict
from typing import List

import datasets
from datasets import load_dataset
from .base import DSTCBase, DSTCBuilderConfig

_URL = "https://github.com/festvox/datasets-CMU_DoG/archive/refs/heads/master.zip"

class CMUDoG(DSTCBase, datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        DSTCBuilderConfig(
            name=name,
            version=datasets.Version("1.0.0"),
            description="",
        ) for name in ["generation", "evaluation"]
    ]

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

    def _samples_from_dialog(self, dialog, doc, grounded_speaker):
        history = []
        out = []
        
        for i, turn in enumerate(dialog["history"]):
            speaker = "S" if turn["uid"] == grounded_speaker else "U"
            history.append({
                "speaker": speaker,
                "text": turn["text"]
            })
            doc_idx = turn["docIdx"]
            content = doc[str(doc_idx)]
            
            def process_value(value):
                if isinstance(value, list):
                    return " ".join(value)
                else:
                    return str(value)
            
            if doc_idx == 0:
                content = " ".join([f"{key}: {process_value(value)}" for key, value in content.items()])

            if speaker == "S":
                sample = { 
                    "target": True,
                    "knowledge": [
                    {
                        "domain": None,
                        "entity_id": None,
                        "doc_id": dialog["wikiDocumentIdx"],
                        "entity_name": doc["name"],
                        "title": None,
                        "body": content,
                        "score": 0.0
                    }
                    ],
                    "response": history[i]["text"],
                    "turns": history[:i],
                    "source": "cmu_dog"
                }
                out.append(sample)

        return out

    def _map_to_dstc_format(self, dialog, doc_data):
        doc_idx = dialog["wikiDocumentIdx"]
        doc = doc_data[doc_idx]
        out = []
        grounded_speakers = dialog["whoSawDoc"]
        
        for speaker in grounded_speakers:
            samples = self._samples_from_dialog(dialog, doc, speaker)
            out.extend(samples)

        return out

    def _get_document_data(self, base_path):
        data = {}
        doc_path = os.path.join(base_path, "WikiData")
        
        for file in os.listdir(doc_path):
            file_path = os.path.join(doc_path, file)
        
            with open(file_path, "r") as f:
                content = json.load(f)
                idx = content["wikiDocumentIdx"]
                data[idx] = content
                data[idx]["name"] = file.split(".")[0]
        
        return data

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        url_to_download = _URL
        file_path = self._download_files(url_to_download, self.config.data_files, dl_manager)
        base_path = os.path.join(file_path, "datasets-CMU_DoG-master")
        doc_data = self._get_document_data(base_path)

        splits = ["train", "valid", "test"]
        data = {split: [] for split in splits}
        for split in splits:
            directory = os.path.join(base_path, "Conversations", split)
            for file in os.listdir(directory):
                file_name = os.path.join(directory, file)
                with open(file_name, "r") as f:
                    content = json.load(f)
                    dialog = self._map_to_dstc_format(content, doc_data)
                    data[split].extend(dialog)

        return [
            datasets.SplitGenerator(
                name=ds_split, gen_kwargs={
                    "data": data[split],
                })
            for ds_split, split in zip([datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST], splits)
        ]

    def _generate_examples(self, data):

        for idx, sample in enumerate(data):
            sample["id"] = str(idx)
            yield idx, sample
