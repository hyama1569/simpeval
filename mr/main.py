#from cProfile import label
import os
from random import shuffle
#import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
#from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
from pytorch_lightning.strategies.ddp import DDPStrategy

checkpoint_path = './checkpoints/mr_1'
data_file_name='augmented_wikiauto_max_cnt_6_randomsamp_16_labeled_by_exp_1.pickle'
n_classes=1
n_linears=1
d_hidden_linear=128
dropout_rate=0.2
max_token_len=512
pl_seed=1
test_size=0.3
valid_size=0.3
n_epochs=15
batch_size=256
learning_rate=0.05
n_gpus=4
orig_column_name='original'
simp_column_name='simple'
label_column_name='label'
case_num_column_name='case_number'
patience=5
patience_min_delta=0

class AugmentedDataset(Dataset):
    def __init__(
        self, 
        data: pd.DataFrame, 
        tokenizer: BertTokenizer, 
        max_token_len: int,
        orig_column_name: str,
        simp_column_name: str,
        label_column_name: str,
        case_num_column_name:str,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.orig_column_name = orig_column_name
        self.simp_column_name = simp_column_name
        self.label_column_name = label_column_name
        self.case_num_column_name = case_num_column_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        origs = data_row[self.orig_column_name]
        simps = data_row[self.simp_column_name]
        labels = data_row[self.label_column_name]
        case_nums = data_row[self.case_num_column_name]

        encoding_origs = self.tokenizer.encode_plus(
            origs,
            add_special_tokens=True,
            max_length=self.max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        encoding_simps = self.tokenizer.encode_plus(
            simps,
            add_special_tokens=True,
            max_length=self.max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return dict(
            origs=dict(
                input_ids=encoding_origs["input_ids"].flatten(),
                attention_mask=encoding_origs["attention_mask"].flatten(),
            ),
            simps=dict(
                input_ids=encoding_simps["input_ids"].flatten(),
                attention_mask=encoding_simps["attention_mask"].flatten(),
            ),
            labels=torch.tensor(labels),
            case_nums=torch.tensor(case_nums)
        )

class CreateDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame, 
        valid_df: pd.DataFrame, 
        test_df: pd.DataFrame, 
        batch_size: int, 
        max_token_len: int, 
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
              self.orig_column_name,
              self.simp_column_name,
              self.label_column_name,
              self.case_num_column_name,
            )
          self.vaild_dataset = AugmentedDataset(
              self.valid_df, 
              self.tokenizer, 
              self.max_token_len,
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
        pretrained_model='bert-base-uncased',
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model, output_hidden_states=True, return_dict=True)
        
        if n_linears == 1:
            self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
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

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask=attention_mask)
        preds = self.classifier(output.pooler_output)
        preds = torch.flatten(preds)
        return preds, output
      
    def training_step(self, batch, batch_idx):
        orig_preds, output = self.forward(input_ids=batch["origs"]["input_ids"],
                                    attention_mask=batch["origs"]["attention_mask"])
        simp_preds, output = self.forward(input_ids=batch["simps"]["input_ids"],
                                    attention_mask=batch["simps"]["attention_mask"])
        loss = self.criterion(simp_preds, orig_preds, batch["labels"])
        return {'loss': loss,
                'batch_preds': [orig_preds, simp_preds],
                'batch_labels': batch["labels"]}

    def validation_step(self, batch, batch_idx):
        orig_preds, output = self.forward(input_ids=batch["origs"]["input_ids"],
                                    attention_mask=batch["origs"]["attention_mask"])
        simp_preds, output = self.forward(input_ids=batch["simps"]["input_ids"],
                                    attention_mask=batch["simps"]["attention_mask"])
        loss = self.criterion(simp_preds, orig_preds, batch["labels"])
        return {'loss': loss,
                'batch_preds': [orig_preds, simp_preds],
                'batch_labels': batch["labels"]}

    def test_step(self, batch, batch_idx):
        orig_preds, output = self.forward(input_ids=batch["origs"]["input_ids"],
                                    attention_mask=batch["origs"]["attention_mask"])
        simp_preds, output = self.forward(input_ids=batch["simps"]["input_ids"],
                                    attention_mask=batch["simps"]["attention_mask"])
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
                     

    def validation_epoch_end(self, outputs, mode="val"):
        epoch_orig_preds = torch.cat([x['batch_preds'][0] for x in outputs])
        epoch_simp_preds = torch.cat([x['batch_preds'][1] for x in outputs])
        epoch_labels = torch.cat([x['batch_labels'] for x in outputs])
        epoch_loss = self.criterion(epoch_simp_preds, epoch_orig_preds, epoch_labels)
        self.log(f"{mode}_loss", epoch_loss, logger=True)                  

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

#@hydra.main(config_path=".", config_name="config")
def main():
    pl.seed_everything(pl_seed, workers=True)
    #cwd = hydra.utils.get_original_cwd()
    #wandb_logger = WandbLogger(
    #    name=("exp_" + str(cfg.wandb.exp_num)),
    #    project=cfg.wandb.project,
    #    tags=cfg.wandb.tags,
    #    log_model=True,
    #)
    #checkpoint_path = os.path.join(
    #    wandb_logger.experiment.dir, cfg.path.checkpoint_path
    #)
    #wandb_logger.log_hyperparams(cfg)
    data = pd.read_pickle(data_file_name)
    train, test = train_test_split(data, test_size=test_size, shuffle=True)
    train, valid = train_test_split(train, test_size=valid_size, shuffle=True)
    data_module = CreateDataModule(
        train_df=train,
        valid_df=valid,
        test_df=test,
        batch_size=batch_size,
        max_token_len=max_token_len,
    )
    data_module.setup(stage='fit')

    call_backs = make_callbacks(
        patience_min_delta, patience, checkpoint_path
    )
    model = BertRanker(
        n_classes=n_classes,
        n_linears=n_linears,
        d_hidden_linear=d_hidden_linear,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
    )
    trainer = pl.Trainer(
        max_epochs=n_epochs,
        devices=n_gpus,
        accelerator="gpu",
        #progress_bar_refresh_rate=30,
        callbacks=call_backs,
        strategy=DDPStrategy(find_unused_parameters=False),
        deterministic=True
    )
    trainer.fit(model, data_module)
                           
    data_module.setup(stage='test')                       
    results = trainer.test(ckpt_path=call_backs[1].best_model_path, datamodule=data_module)                      
    print(results)

if __name__ == "__main__":
    main()
