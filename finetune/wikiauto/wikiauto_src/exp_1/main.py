import os
from random import shuffle

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
        train_df: pd.DataFrame=None, 
        valid_df: pd.DataFrame=None, 
        test_df: pd.DataFrame=None, 
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
                classifier.add_module(nn.Linear(d_hidden_linear, d_hidden_linear))
                classifier.add_module(nn.Sigmoid())
                classifier.add_module(nn.Dropout(p=dropout_rate))
                classifier.add_module(nn.Linear(d_hidden_linear, n_classes))
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
      
    def training_step(self, batch, batch_idx):
        loss, preds, output = self.forward(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"],
                                    labels=batch["labels"])
        return {'loss': loss,
                'batch_preds': preds,
                'batch_labels': batch["labels"]}

    def validation_step(self, batch, batch_idx):
        loss, preds, output = self.forward(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"],
                                    labels=batch["labels"])
        return {'loss': loss,
                'batch_preds': preds,
                'batch_labels': batch["labels"]}

    def test_step(self, batch, batch_idx):
        loss, preds, output = self.forward(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"],
                                    labels=batch["labels"])
        return {'loss': loss,
                'batch_preds': preds,
                'batch_labels': batch["labels"]}
    
    def training_epoch_end(self, outputs, mode="train"):
        epoch_preds = torch.cat([x['batch_preds'] for x in outputs])
        epoch_labels = torch.cat([x['batch_labels'] for x in outputs])
        epoch_loss = self.criterion(epoch_preds, epoch_labels)
        self.log(f"{mode}_loss", epoch_loss, logger=True)

        num_correct = (epoch_preds.argmax(dim=1) == epoch_labels).sum().item()
        epoch_accuracy = num_correct / len(epoch_labels)
        self.log(f"{mode}_accuracy", epoch_accuracy, logger=True)
        
        epoch_auroc = auroc(epoch_preds, epoch_labels, num_classes=self.n_classes)
        self.log(f"{mode}_auroc", epoch_auroc)                   

    def validation_epoch_end(self, outputs, mode="val"):
        epoch_preds = torch.cat([x['batch_preds'] for x in outputs])
        epoch_labels = torch.cat([x['batch_labels'] for x in outputs])
        epoch_loss = self.criterion(epoch_preds, epoch_labels)
        self.log(f"{mode}_loss", epoch_loss, logger=True)

        num_correct = (epoch_preds.argmax(dim=1) == epoch_labels).sum().item()
        epoch_accuracy = num_correct / len(epoch_labels)
        self.log(f"{mode}_accuracy", epoch_accuracy, logger=True)
                           
        epoch_auroc = auroc(epoch_preds, epoch_labels, num_classes=self.n_classes)
        self.log(f"{mode}_auroc", epoch_auroc)                   

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
    #df = pd.read_csv(cfg.path.data_file_name, sep="\t").dropna().reset_index(drop=True)
    #df[cfg.training.label_column_name] = np.argmax(df.iloc[:, 2:].values, axis=1)
    #df[[cfg.training.text_column_name, cfg.training.label_column_name]]
    data = pd.read_pickle(cfg.path.data_file_name)
    #train_df, test_df = train_test_split(data, test_size=0.2, random_state=0)
    #train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=0)
    #data_module = CreateDataModule(train_df=train_df, valid_df=valid_df, test_df=test_df)
    #data_module.setup(stage=None)
    train, test = train_test_split(data, test_size=cfg.training.test_size, shuffle=True)
    train, valid = train_test_split(train, test_size=cfg.training.valid_size, shuffle=True)
    data_module = CreateDataModule(
        train,
        valid,
        test,
        cfg.training.batch_size,
        cfg.model.max_token_len,
    )
    data_module.setup(stage='fit')

    call_backs = make_callbacks(
        cfg.callbacks.patience_min_delta, cfg.callbacks.patience, checkpoint_path
    )
    model = BertClassifier(
        n_classes=cfg.model.n_classes,
        n_linears=cfg.model.n_linears,
        d_hidden_linear=cfg.model.d_hidden_linear,
        dropout_rate=cfg.model.dropout_rate,
        learning_rate=cfg.training.learning_rate,
    )
    trainer = pl.Trainer(
        max_epochs=cfg.training.n_epochs,
        gpus=1,
        #progress_bar_refresh_rate=30,
        callbacks=call_backs,
        logger=wandb_logger,
    )
    trainer.fit(model, data_module)
                           
    data_module.setup(stage='test')                       
    results = trainer.test(ckpt_path=call_backs[1].best_model_path, datamodule=data_module)                      
    print(results)

if __name__ == "__main__":
    main()
