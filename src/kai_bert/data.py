from datasets import Dataset
from typing import Union
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, AutoTokenizer
from pytorch_lightning import LightningDataModule


class LanguageModelingDataModule(LightningDataModule):
    
    train_data: Dataset
    valid_data: Dataset
    test_data: Dataset
    
    def __init__(
        self,
        tokenizer: Union[str, AutoTokenizer],
        train_file_path: str,
        valid_file_path: str,
        test_file_path: str,
        text_column_name: str = 'text',
        batch_size: int = 32,
        num_proc: int = 4,
        **kwargs
    ):
        super().__init__()
        
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
        
        assert self.tokenizer.mask_token is not None, "Tokenizer mask token is missing. Ensure the tokenizer supports masked language modeling."

        self.train_file = train_file_path
        self.valid_file = valid_file_path
        self.test_file = test_file_path
        self.text_column_name = text_column_name
        self.batch_size = batch_size
        self.num_proc = num_proc
        self.csv_kwargs = kwargs
        
        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer
        )
    
    
    def _process_text(self, examples):
        # TODO(khairi): handle when max_length is set
        return self.tokenizer(
            examples[self.text_column_name],
            return_tensors='pt',
            padding=True
        )
    
    def _load_dataset(self, input_file: str) -> Dataset:
        return Dataset.from_csv(input_file, **self.csv_kwargs) \
            .map(self._process_text, batched=True, batch_size=self.batch_size, num_proc=self.num_proc) \
            .select_columns(['input_ids', 'attention_mask']) \
            .with_format('torch')
    
    def _get_dataloader(self, dataset: Dataset, shuffle: bool = False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_proc,
            collate_fn=self.data_collator
        )
    
    def setup(self, stage=None):
        self.train_data = self._load_dataset(self.train_file)
        self.valid_data = self._load_dataset(self.valid_file)
        self.test_data = self._load_dataset(self.test_file)

    def train_dataloader(self):
        return self._get_dataloader(self.train_data, shuffle=True)
    
    def val_dataloader(self):
        return self._get_dataloader(self.valid_data, shuffle=True)
    
    def test_dataloader(self):
        return self._get_dataloader(self.test_data, shuffle=False)
    
    def __repr__(self):
        return f"""
        LanguageModelingDataModule(
            tokenizer={self.tokenizer.__repr__()},
            train_file={self.train_file.__repr__()},
            valid_file={self.valid_file.__repr__()},
            test_file={self.test_file.__repr__()},
            text_column_name={self.text_column_name.__repr__()},
            batch_size={self.batch_size.__repr__()},
            num_proc={self.num_proc.__repr__()},
            csv_kwargs={self.csv_kwargs.__repr__()}
            data_collator={self.data_collator.__repr__()}
        )
    """