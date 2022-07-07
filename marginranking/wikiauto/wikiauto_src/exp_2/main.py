from ast import Add
from cProfile import label
import os
from random import shuffle
from eo_augmentation.utils_neural_jacana import preprocess_texts
import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import accuracy, auroc
from transformers import BertModel, BertTokenizer
from tseval.feature_extraction import get_compression_ratio, get_wordrank_score
import sys
sys.path.append('../../../eo_augmentaion')
from utils_neural_jacana import *
from utils_apply_operations import *
from utils_extract_edit_operation import *
import argparse
import nltk
nltk.download('punkt')
import spacy

class AdditionalFeatureExtractor():
    def __init__(
        self,
        origin: str,
        sent: str,
    ):
        self.origin = origin
        self.sent = sent
    
    def wordrank_score(self):
        return get_wordrank_score(self.sent)
    
    def maximum_deptree_depth(self):
        def tree_height(root):
            if not list(root.children):
                return 1
            else:
                return 1 + max(tree_height(x) for x in root.children)
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(self.sent)
        roots = [sent.root for sent in doc.sents]
        return max([tree_height(root) for root in roots])

    def compression_ratio(self):
        return get_compression_ratio(self.origin, self.sent)
    
    def edit_operation_nums(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--batchsize", default=1, type=int)
        parser.add_argument("--learning_rate", default=1e-5, type=float)
        parser.add_argument("--max_epoch", default=6, type=int)
        parser.add_argument("--max_span_size", default=1, type=int)
        parser.add_argument("--max_seq_length", default=128, type=int)
        parser.add_argument("--max_sent_length", default=70, type=int)
        parser.add_argument("--seed", default=1234, type=int)
        parser.add_argument("--dataset", default='mtref', type=str)
        parser.add_argument("--sure_and_possible", default='True', type=str)
        parser.add_argument("--distance_embedding_size", default=128, type=int)
        parser.add_argument("--use_transition_layer", default='False', type=str, help='if False, will set transition score to 0.')
        parser.add_argument("batch_size", default=1, type=int)
        parser.add_argument("my_device", default='cuda', type=str)
        args = parser.parse_args(args=[])
        model = prepare_model(args)
        aligns = get_alignment(model, args, self.origin, self.sent)

        sent1_toks = preprocess_texts([self.origin])
        sent2_toks = preprocess_texts([self.sent])
        edit_sequences = get_edit_sequences(sent1_toks, sent2_toks, aligns)
        return len(edit_sequences)

class AugmentedDataset(Dataset):
    def __init__(
        self, 
        data: pd.DataFrame, 
        tokenizer: BertTokenizer, 
        max_token_len: int,
        origin_column_name: str,
        orig_column_name: str,
        simp_column_name: str,
        label_column_name: str,
        case_num_column_name:str,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.origin_column_name = origin_column_name
        self.orig_column_name = orig_column_name
        self.simp_column_name = simp_column_name
        self.label_column_name = label_column_name
        self.case_num_column_name = case_num_column_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        origin = data_row[self.origin_column_name]
        orig = data_row[self.orig_column_name]
        simp = data_row[self.simp_column_name]
        label = data_row[self.label_column_name]
        case_num = data_row[self.case_num_column_name]

        encoding_origs = self.tokenizer.encode_plus(
            orig,
            add_special_tokens=True,
            max_length=self.max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        origs_feature_ext = AdditionalFeatureExtractor(origin=origin, sent=orig)
        origs_added_features = dict(
            wordrank_score=origs_feature_ext.wordrank_score(),
            maximum_deptree_depth=origs_feature_ext.maximum_deptree_depth(),
            compression_ratio=origs_feature_ext.compression_ratio(),
            edit_operation_nums=origs_feature_ext.edit_operation_nums(),
        )
        origs_added_features_tensor = torch.tensor([list(origs_added_features.values())])

        encoding_simps = self.tokenizer.encode_plus(
            simp,
            add_special_tokens=True,
            max_length=self.max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        simps_feature_ext = AdditionalFeatureExtractor(origin=origin, sent=simp)
        simps_added_features = dict(
            wordrank_score=simps_feature_ext.wordrank_score(),
            maximum_deptree_depth=simps_feature_ext.maximum_deptree_depth(),
            compression_ratio=simps_feature_ext.compression_ratio(),
            edit_operation_nums=simps_feature_ext.edit_operation_nums(),
        )
        simps_added_features_tensor = torch.tensor([list(simps_added_features.values())])

        return dict(
            origs=dict(
                input_ids=encoding_origs["input_ids"].flatten(),
                attention_mask=encoding_origs["attention_mask"].flatten(),
                added_features=origs_added_features_tensor.flatten(),
            ),
            simps=dict(
                input_ids=encoding_simps["input_ids"].flatten(),
                attention_mask=encoding_simps["attention_mask"].flatten(),
                added_features=simps_added_features_tensor.flatten(),
            ),
            labels=torch.tensor(label),
            case_nums=torch.tensor(case_num),
        )

class CreateDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame, 
        valid_df: pd.DataFrame, 
        test_df: pd.DataFrame, 
        batch_size: int, 
        max_token_len: int, 
        origin_column_name: str = 'origin',
        orig_column_name: str = 'original',
        simp_column_name: str = 'simple',
        label_column_name: str = 'label',
        case_num_column_name: str = 'case_number',
        pretrained_model='bert-base-uncased',
    ):
        super().__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.origin_column_name = origin_column_name
        self.orig_column_name = orig_column_name
        self.simp_column_name = simp_column_name
        self.label_column_name = label_column_name
        self.case_num_column_name = case_num_column_name
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)

    def setup(self, stage):
        if stage == "fit":
          self.train_dataset = AugmentedDataset(
              self.train_df, 
              self.tokenizer, 
              self.max_token_len,
              self.origin_column_name,
              self.orig_column_name,
              self.simp_column_name,
              self.label_column_name,
              self.case_num_column_name,
            )
          self.vaild_dataset = AugmentedDataset(
              self.valid_df, 
              self.tokenizer, 
              self.max_token_len,
              self.origin_column_name,
              self.orig_column_name,
              self.simp_column_name,
              self.label_column_name,
              self.case_num_column_name,
            )
        if stage == "test":
          self.test_dataset = AugmentedDataset(
              self.test_df, 
              self.tokenizer, 
              self.max_token_len,
              self.origin_column_name,
              self.orig_column_name,
              self.simp_column_name,
              self.label_column_name,
              self.case_num_column_name,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count())

    def val_dataloader(self):
        return DataLoader(self.vaild_dataset, batch_size=self.batch_size, num_workers=os.cpu_count())

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=os.cpu_count())

class BertRanker(pl.LightningModule):
    def __init__(
        self, 
        n_classes: int, 
        n_linears: int,
        d_hidden_linear: int,
        dropout_rate: float,
        learning_rate: float,
        added_feature_num: int,
        pretrained_model='bert-base-uncased',
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model, output_hidden_states=True, return_dict=True)
        
        if n_linears == 1:
            self.classifier = nn.Linear(self.bert.config.hidden_size + added_feature_num, n_classes)
        else:
            classifier = nn.Sequential(
                nn.Linear(self.bert.config.hidden_size, d_hidden_linear),
                nn.Sigmoid(),
                nn.Dropout(p=dropout_rate),
            )
            for i in range(n_linears-1):
                classifier.add_module('fc_{}'.format(i), nn.Linear(d_hidden_linear, d_hidden_linear))
                classifier.add_module('activate_{}'.format(i), nn.Sigmoid())
                classifier.add_module('dropout_{}'.format(i), nn.Dropout(p=dropout_rate))
            classifier.add_module('fc_last', nn.Linear(d_hidden_linear, n_classes))
            self.classifier = classifier
        
        self.lr = learning_rate
        self.criterion = nn.MarginRankingLoss(margin=1.0)
        self.n_classes = n_classes

        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask, added_features):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = torch.cat([added_features, output.pooler_output], dim=1)
        preds = self.classifier(output)
        preds = torch.flatten(preds)
        return preds, output
      
    def training_step(self, batch, batch_idx):
        orig_preds, output = self.forward(input_ids=batch["origs"]["input_ids"],
                                    attention_mask=batch["origs"]["attention_mask"],
                                    added_features=batch["origs"]["added_features"])
        simp_preds, output = self.forward(input_ids=batch["simps"]["input_ids"],
                                    attention_mask=batch["simps"]["attention_mask"],
                                    added_features=batch["simps"]["added_features"])
        loss = self.criterion(simp_preds, orig_preds, batch["labels"])
        return {'loss': loss,
                'batch_preds': [orig_preds, simp_preds],
                'batch_labels': batch["labels"]}

    def validation_step(self, batch, batch_idx):
        orig_preds, output = self.forward(input_ids=batch["origs"]["input_ids"],
                                    attention_mask=batch["origs"]["attention_mask"],
                                    added_features=batch["origs"]["added_features"])
        simp_preds, output = self.forward(input_ids=batch["simps"]["input_ids"],
                                    attention_mask=batch["simps"]["attention_mask"],
                                    added_features=batch["simps"]["added_features"])
        loss = self.criterion(simp_preds, orig_preds, batch["labels"])
        return {'loss': loss,
                'batch_preds': [orig_preds, simp_preds],
                'batch_labels': batch["labels"]}

    def test_step(self, batch, batch_idx):
        orig_preds, output = self.forward(input_ids=batch["origs"]["input_ids"],
                                    attention_mask=batch["origs"]["attention_mask"],
                                    added_features=batch["origs"]["added_features"])
        simp_preds, output = self.forward(input_ids=batch["simps"]["input_ids"],
                                    attention_mask=batch["simps"]["attention_mask"],
                                    added_features=batch["simps"]["added_features"])
        loss = self.criterion(simp_preds, orig_preds, batch["labels"])
        return {'loss': loss,
                'batch_preds': [orig_preds, simp_preds],
                'batch_labels': batch["labels"]}
    
    def training_epoch_end(self, outputs, mode="train"):
        epoch_orig_preds = torch.cat([x['batch_preds'][0] for x in outputs])
        epoch_simp_preds = torch.cat([x['batch_preds'][1] for x in outputs])
        epoch_labels = torch.cat([x['batch_labels'] for x in outputs])
        epoch_loss = self.criterion(epoch_simp_preds, epoch_orig_preds, epoch_labels)
        self.log(f"{mode}_loss", epoch_loss, logger=True)

        num_correct = (torch.sign(epoch_simp_preds-epoch_orig_preds) == epoch_labels).sum().item()
        epoch_accuracy = num_correct / len(epoch_labels)
        self.log(f"{mode}_rank_accuracy", epoch_accuracy, logger=True)                   

    def validation_epoch_end(self, outputs, mode="val"):
        epoch_orig_preds = torch.cat([x['batch_preds'][0] for x in outputs])
        epoch_simp_preds = torch.cat([x['batch_preds'][1] for x in outputs])
        epoch_labels = torch.cat([x['batch_labels'] for x in outputs])
        epoch_loss = self.criterion(epoch_simp_preds, epoch_orig_preds, epoch_labels)
        self.log(f"{mode}_loss", epoch_loss, logger=True)

        num_correct = (torch.sign(epoch_simp_preds-epoch_orig_preds) == epoch_labels).sum().item()
        epoch_accuracy = num_correct / len(epoch_labels)
        self.log(f"{mode}_rank_accuracy", epoch_accuracy, logger=True)                      

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

def make_callbacks(min_delta, patience, checkpoint_path):

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="{epoch}",
        save_top_k=3,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=min_delta, patience=patience, mode="min"
    )

    return [early_stop_callback, checkpoint_callback]

@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.training.pl_seed, workers=True)
    cwd = hydra.utils.get_original_cwd()
    wandb_logger = WandbLogger(
        name=("exp_" + str(cfg.wandb.exp_num)),
        project=cfg.wandb.project,
        tags=cfg.wandb.tags,
        log_model=True,
    )
    checkpoint_path = os.path.join(
        wandb_logger.experiment.dir, cfg.path.checkpoint_path
    )
    wandb_logger.log_hyperparams(cfg)
    data = pd.read_pickle(cfg.path.data_file_name)
    train, test = train_test_split(data, test_size=cfg.training.test_size, shuffle=True)
    train, valid = train_test_split(train, test_size=cfg.training.valid_size, shuffle=True)
    data_module = CreateDataModule(
        train_df=train,
        valid_df=valid,
        test_df=test,
        batch_size=cfg.training.batch_size,
        max_token_len=cfg.model.max_token_len,
    )
    data_module.setup(stage='fit')

    call_backs = make_callbacks(
        cfg.callbacks.patience_min_delta, cfg.callbacks.patience, checkpoint_path
    )
    model = BertRanker(
        n_classes=cfg.model.n_classes,
        n_linears=cfg.model.n_linears,
        d_hidden_linear=cfg.model.d_hidden_linear,
        dropout_rate=cfg.model.dropout_rate,
        learning_rate=cfg.training.learning_rate,
        added_feature_num=cfg.model.added_feature_num,    
    )
    trainer = pl.Trainer(
        max_epochs=cfg.training.n_epochs,
        devices=cfg.training.n_gpus,
        accelerator="gpu",
        #progress_bar_refresh_rate=30,
        callbacks=call_backs,
        logger=wandb_logger,
        deterministic=True
    )
    trainer.fit(model, data_module)
                           
    data_module.setup(stage='test')                       
    results = trainer.test(ckpt_path=call_backs[1].best_model_path, datamodule=data_module)                      
    print(results)

if __name__ == "__main__":
    main()
