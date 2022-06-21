import pickle
import json
import pandas as pd
import os
import random
import tqdm
import itertools
from transformers import BertTokenizer, BertModel
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torch
from typing import List
import hydra
from omegaconf import DictConfig

class CreateDataset(Dataset):
    def __init__(
        self, 
        data: pd.DataFrame, 
        tokenizer: BertTokenizer, 
        max_token_len: int,
        orig_column_name: str,
        simp_column_name: str,
        label_column_name: str,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.orig_column_name = orig_column_name
        self.simp_column_name = simp_column_name
        self.label_column_name = label_column_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        origs = data_row[self.orig_column_name]
        simps = data_row[self.simp_column_name]
        labels = data_row[self.label_column_name]

        encoding = self.tokenizer.encode_plus(
            origs,
            simps,
            add_special_tokens=True,
            max_length=self.max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return dict(
            origs=origs,
            simps=simps,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.tensor(labels)
        )

class CreateDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int, 
        max_token_len: int, 
        train_df: pd.DataFrame = None, 
        valid_df: pd.DataFrame = None, 
        test_df: pd.DataFrame = None, 
        orig_column_name: str = 'original',
        simp_column_name: str = 'simple',
        label_column_name: str = 'label',
        pretrained_model='bert-base-uncased',
    ):
        super().__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.orig_column_name = orig_column_name
        self.simp_column_name = simp_column_name
        self.label_column_name = label_column_name
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)

    def setup(self, stage):
        if stage == "fit":
          self.train_dataset = CreateDataset(
              self.train_df, 
              self.tokenizer, 
              self.max_token_len,
              self.orig_column_name,
              self.simp_column_name,
              self.label_column_name,
            )
          self.vaild_dataset = CreateDataset(
              self.valid_df, 
              self.tokenizer, 
              self.max_token_len,
              self.orig_column_name,
              self.simp_column_name,
              self.label_column_name, 
            )
        if stage == "test":
          self.test_dataset = CreateDataset(
              self.test_df, 
              self.tokenizer, 
              self.max_token_len,
              self.orig_column_name,
              self.simp_column_name,
              self.label_column_name,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count())

    def val_dataloader(self):
        return DataLoader(self.vaild_dataset, batch_size=self.batch_size, num_workers=os.cpu_count())

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=os.cpu_count())

class BertClassifier(pl.LightningModule):
    def __init__(
        self, 
        n_classes: int, 
        n_linears: int,
        d_hidden_linear: int,
        dropout_rate: float,
        learning_rate: float=None,
        pretrained_model='bert-base-uncased',
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model, output_hidden_states=True, return_dict=True)
        #self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        if n_linears == 1:
            self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        else:
            classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, d_hidden_linear),
            nn.Sigmoid(),
            nn.Dropout(p=dropout_rate)
        )
            for i in range(n_linears-1):
                    classifier.add_module('fc_{}'.format(i), nn.Linear(d_hidden_linear, d_hidden_linear))
                    classifier.add_module('activate_{}'.format(i), nn.Sigmoid())
                    classifier.add_module('dropout_{}'.format(i), nn.Dropout(p=dropout_rate))
            classifier.add_module('fc_last', nn.Linear(d_hidden_linear, n_classes))
            self.classifier = classifier
        self.lr = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.n_classes = n_classes

        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        preds = self.classifier(output.pooler_output)
        loss = 0
        if labels is not None:
            loss = self.criterion(preds, labels)
        return loss, preds, output
        
def random_sample_augmented_data(
    sources: List[str], 
    targets: List[str], 
    aug_data: List[str], 
    n_samples: int,

) -> pd.DataFrame:
    sampled_sources = []
    sampled_targets = []
    sampled_case_nums = []
    sampled_labels = []
    sampled_sources_unlabeled = []
    sampled_targets_unlabeled = []
    sampled_case_nums_unlabeled = []
    case_num = 0
    print("start random sampling data.")
    for i in tqdm.tqdm(range(len(aug_data))):
        if sources[i] == targets[i]:
            continue
        else:
            sampled_sources.append(sources[i])
            sampled_targets.append(targets[i])
            sampled_case_nums.append(case_num)
            sampled_labels.append(1)
            sampled_sources.append(targets[i])
            sampled_targets.append(sources[i])
            sampled_case_nums.append(case_num)
            sampled_labels.append(-1)
            paired_cands = []
            for perm in itertools.permutations(aug_data[i]+[sources[i], targets[i]], 2):
                if (perm[0] == sources[i] and perm[1] == targets[i]) or (perm[0] == targets[i] and perm[1] == sources[i]):
                    continue
                elif perm[0] == sources[i]:
                    paired_cands.append([perm[0], perm[1], 1])
                elif perm[1] == sources[i]:
                    paired_cands.append([perm[0], perm[1], -1])
                elif perm[0] == targets[i]:
                    paired_cands.append([perm[0], perm[1], -1])
                elif perm[1] == targets[i]:
                    paired_cands.append([perm[0], perm[1], 1])
                else:
                    paired_cands.append([perm[0], perm[1], -100]) #dummy label
        rs_paired_cands = random.sample(paired_cands, min(n_samples, len(paired_cands)))
        for i in range(len(rs_paired_cands)):
            if rs_paired_cands[i][2] == 1 or rs_paired_cands[i][2] == -1:
                sampled_sources.append(rs_paired_cands[i][0])
                sampled_targets.append(rs_paired_cands[i][1])
                sampled_case_nums.append(case_num)
                sampled_labels.append(rs_paired_cands[i][2])
            else:
                sampled_sources_unlabeled.append(rs_paired_cands[i][0]) 
                sampled_targets_unlabeled.append(rs_paired_cands[i][1])
                sampled_case_nums_unlabeled.append(case_num)
        case_num += 1
    random_sampled_df_labeled = pd.DataFrame({'original':sampled_sources, 'simple':sampled_targets, 'case_number':sampled_case_nums, 'label':sampled_labels})
    random_sampled_df_unlabeled = pd.DataFrame({'original':sampled_sources_unlabeled, 'simple':sampled_targets_unlabeled, 'case_number':sampled_case_nums_unlabeled})
    return random_sampled_df_labeled, random_sampled_df_unlabeled

@hydra.main(config_path="../wikiauto_src/exp_1", config_name="config")
def main(cfg: DictConfig):
    with open(str(cfg.path.aug_data), 'rb') as f:
        aug_data = pickle.load(f)
    with open(str(cfg.path.sources), 'rb') as f:
        sources = pickle.load(f)
    with open(str(cfg.path.targets), 'rb') as f:
        targets = pickle.load(f)

    n_samples = cfg.dataprep.n_random_sample
    random.seed(cfg.dataprep.random_seed)

    random_sampled_df_labeled, random_sampled_df_unlabeled = random_sample_augmented_data(sources, targets, aug_data, n_samples)
    with open(str(cfg.path.random_sampled_data_labeled), 'wb') as f:
        pickle.dump(random_sampled_df_labeled, f)
    with open(str(cfg.path.random_sampled_data_unlabeled), 'wb') as f:
        pickle.dump(random_sampled_df_unlabeled, f)

    data_module = CreateDataModule(cfg.dataprep.batch_size, cfg.dataprep.max_token_len, test_df=random_sampled_df_unlabeled)
    data_module.setup(stage='test')
    test_dataloader = data_module.test_dataloader()
    model = BertClassifier.load_from_checkpoint(
        n_classes=2,
        n_linears=cfg.dataprep.n_linears, 
        d_hidden_linear=cfg.dataprep.d_hidden_linear, 
        dropout_rate=cfg.dataprep.dropout_rate,
        checkpoint_path=str(cfg.path.finetuned),
    )
    #model = nn.DataParallel(model, device_ids=[2,3])
    device = 'cuda'
    model.to(device)

    predicted_labels = []
    with torch.no_grad():
        print("start prediction of fine-tuned model")
        for batch in tqdm.tqdm(test_dataloader):
            input_ids=batch["input_ids"]
            attention_mask=batch["attention_mask"]
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            loss, preds, output = model(input_ids, attention_mask)
            predicted_labels.append(preds)

    raw_predicted_labels = torch.cat([x for x in predicted_labels]).to('cpu')
    random_sampled_df_unlabeled['label'] = raw_predicted_labels
    with open(cfg.path.raw_predicted_path, 'wb') as f:
        pickle.dump(random_sampled_df_unlabeled, f)

    #argmaxed_predicted_labels = [1 if pred == 1 else -1 for pred in raw_predicted_labels.argmax(dim=1)]

    #random_sampled_df_unlabeled['label'] = argmaxed_predicted_labels

    #with open(str(cfg.path.data_file_name), 'wb') as f:
    #    pickle.dump(random_sampled_df_unlabeled, f)

if __name__ == '__main__':
    main()
