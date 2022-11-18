from datasets import Dataset, load_dataset

class DatasetWalker(object):

    def __init__(self, dataset, split, dataset_filter_dict):
        if dataset_filter_dict is not None:
            self.dataset = load_dataset(dataset, "evaluation", split=split, dataset_filter_dict=dataset_filter_dict)
        else:
            self.dataset = load_dataset(dataset, "evaluation", split=split)

    def __iter__(self):
        for sample in self.dataset:
            log = sample.pop("turns")
            label_keys = ["target", "knowledge", "response"]
            if all([key in sample for key in label_keys]):
                label = {
                    k: sample[k] for k in label_keys
                }
                yield (log, label)
            else:
                yield (log, None)