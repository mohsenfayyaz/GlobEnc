from datasets import load_dataset


class TaskLoader:
    def __init__(self, tokenizer, count=None, max_length=128) -> None:
        self.tokenizer = tokenizer
        self.count = "" if count is None else count
        self.max_length = max_length

    def load_task(self, task):
        task_loaders = {
            "mnli": self.__load_mnli,
            "sst2": self.__load_sst2,
            "hatexplain": self.__load_hatexplain
        }
        return task_loaders[task]()

    def __load_mnli(self):
        dataset = load_dataset("glue", "mnli", split=f"validation_mismatched[:{self.count}]")
        return (
            dataset,
            lambda idx: self.tokenizer.encode_plus(dataset[idx]["premise"], dataset[idx]["hypothesis"],
                                                   return_tensors="pt",
                                                   max_length=self.max_length,
                                                   truncation=True)
        )

    def __load_sst2(self):
        dataset = load_dataset("glue", "sst2", split=f"validation[:{self.count}]")
        return (
            dataset,
            lambda idx: self.tokenizer.encode_plus(dataset[idx]["sentence"], return_tensors="pt",
                                                   max_length=self.max_length,
                                                   truncation=True),
        )

    def __load_hatexplain(self):
        def mode(lst):
            return max(set(lst), key=lst.count)

        def update_data(example):
            example["label"] = mode(example["annotators"]["label"])
            example["text"] = " ".join(example["post_tokens"])
            return example

        dataset = load_dataset("hatexplain", split=f"validation[:{self.count}]").map(update_data)
        return (
            dataset,
            lambda idx: self.tokenizer.encode_plus(dataset[idx]["text"], return_tensors="pt",
                                                   max_length=self.max_length,
                                                   truncation=True),
        )
