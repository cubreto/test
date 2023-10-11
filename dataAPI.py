
import logging
import pandas as pd
from datasets import load_from_disk, load_dataset, concatenate_datasets, DatasetDict, Dataset
from sklearn.utils import resample
from typing import Tuple
from collections import Counter

class SanctionsData:

    def __init__(self, data_path):
        self.dataset = {}
        for split in ['train', 'test']:
            self.dataset[split] = load_from_disk(f"{data_path}/{split}")
        self.reorder_features()
        logging.info(f"Data loaded from {data_path}")
        logging.info("Features reordered.")

    def reorder_features(self):
        ordered_features = ['hit_id', 'business_unit', 'match_tag', 'match_value', 'MATCH_WORD',
                   'offset', 'match_text', 'match_pattern_value','ListName', 'guideline_action', 'guideline_reason',
                   'InputName', 'CATEGORY', 'place', 'hit_name', 'alias',
                   'final_list_type', 'final_input_type', 'final_action', 'final_reason']
            
        for split in ['train', 'test']:
            df = self.dataset[split].to_pandas()
            df = df[ordered_features]
            self.dataset[split] = Dataset.from_pandas(df)

    def get_data_by_index(self, split, start_idx, end_idx):
        logging.info(f"Selected {split} data from index '{start_idx}' to index {end_idx}.")
        return self.dataset[split].select(range(start_idx, end_idx))
