
import logging
import argparse
import time
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from dataAPI import SanctionsData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelAPI:

    def __init__(self, model_name, data_path, output_dir='./output'):
        self.model_name = model_name
        self.data = SanctionsData(data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.output_dir = output_dir
        logger.info(f"ModelAPI initialized with model {model_name}")

    def tokenize_data(self, batch):
        # Your tokenization logic here
        pass

    def prepare_datasets(self, start_idx, end_idx, split='train'):
        dataset_slice = self.data.get_data_by_index(split, start_idx, end_idx)
        tokenized_dataset = dataset_slice.map(self.tokenize_data, batched=True)
        return tokenized_dataset

    def train_model(self, train_dataset, training_args):
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset
        )
        
        start_time = time.time()
        trainer.train()
        end_time = time.time()
        duration = (end_time - start_time) / 60.0
        logger.info(f"Training complete in {duration} minutes.")
