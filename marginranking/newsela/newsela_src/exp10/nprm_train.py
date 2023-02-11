import os
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
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer


class AugmentedDataset(Dataset):
    def __init__(
        self, 
        data: pd.DataFrame, 
        tokenizer: BertTokenizer, 
        max_token_len: int,
        orig_column_name: str,
        simp_column_name: str,
        label_column_name: str,
        group_id_column_name: str,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.orig_column_name = orig_column_name
        self.simp_column_name = simp_column_name
        self.label_column_name = label_column_name
        self.group_id_column_name = group_id_column_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        orig = data_row[self.orig_column_name]
        simp = data_row[self.simp_column_name]
        label = data_row[self.label_column_name]
        group_id = data_row[self.group_id_column_name]

        enc = self.tokenizer.encode_plus(orig, simp, add_special_tokens=True, max_length=self.max_token_len, padding="max_length", truncation=True, return_attention_mask=True, return_tensors='pt')

        return dict(
            encs=dict(
                input_ids=enc["input_ids"].flatten(),
                attention_mask=enc["attention_mask"].flatten(),                     
            ),
            labels=torch.tensor(label),
            group_ids=torch.tensor(group_id),
        )

class CreateDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame = None, 
        valid_df: pd.DataFrame = None, 
        test_df: pd.DataFrame = None, 
        batch_size: int = None, 
        max_token_len: int = None, 
        orig_column_name: str = 'orig',
        simp_column_name: str = 'simp',
        label_column_name: str = 'label',
        group_id_column_name: str = 'group_id',
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
        self.group_id_column_name = group_id_column_name
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
              self.group_id_column_name,
            )
          self.vaild_dataset = AugmentedDataset(
              self.valid_df, 
              self.tokenizer, 
              self.max_token_len,
              self.orig_column_name,
              self.simp_column_name,
              self.label_column_name,
              self.group_id_column_name,
            )
        if stage == "test":
          self.test_dataset = AugmentedDataset(
              self.test_df, 
              self.tokenizer, 
              self.max_token_len,
              self.orig_column_name,
              self.simp_column_name,
              self.label_column_name,
              self.group_id_column_name,
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
        learning_rate: float,
        pooling_type: str,
        added_feature_num: int,
        pretrained_model='bert-base-uncased',
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model, output_hidden_states=True, return_dict=True)
        classifier_hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(classifier_hidden_size + added_feature_num, n_classes)
        
        self.lr = learning_rate
        #self.criterion = nn.MarginRankingLoss(margin=1.0)
        self.criterion = nn.CrossEntropyLoss()
        self.n_classes = n_classes
        self.pooling_type = pooling_type

        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask=attention_mask)
        if self.pooling_type == 'cls':
            cls = output.pooler_output
            #output = torch.cat([added_features.float(), cls], dim=1)
            preds = self.classifier(cls)
        #preds = torch.flatten(preds)
        return preds, output

    def training_step(self, batch, batch_idx, mode="train"):
        preds, _ = self.forward(input_ids=batch["encs"]["input_ids"], attention_mask=batch["encs"]["attention_mask"])
        loss = self.criterion(preds, batch["labels"])
        self.log(f"{mode}_step_loss", loss, logger=True)
        return {'loss': loss,
                #'batch_preds': [orig_preds, simp_preds],
                'batch_preds': preds,
                'batch_labels': batch["labels"]}
    
    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, mode="val")

    def test_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, mode="test")
    
    def training_epoch_end(self, outputs, mode="train"):
        #epoch_orig_preds = torch.cat([x['batch_preds'][0] for x in outputs])
        #epoch_simp_preds = torch.cat([x['batch_preds'][1] for x in outputs])
        epoch_preds = torch.cat([x['batch_preds'] for x in outputs])
        epoch_labels = torch.cat([x['batch_labels'] for x in outputs])
        epoch_loss = self.criterion(epoch_preds, epoch_labels)
        self.log(f"{mode}_loss", epoch_loss, logger=True)

    def validation_epoch_end(self, outputs, mode="val"):
        return self.training_epoch_end(outputs, mode)

    def test_epoch_end(self, outputs, mode="test"):
        return self.training_epoch_end(outputs, mode)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)

def make_callbacks(min_delta, patience, checkpoint_path):
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="{epoch}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=min_delta, patience=patience, mode="min"
    )
    return [early_stop_callback, checkpoint_callback]

@hydra.main(config_path=".", config_name="config_train")
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
    #data['label'] = 1
    #data.rename(columns={'comp':'original', 'simp':'simple'}, inplace=True)
    splitter = GroupShuffleSplit(test_size=cfg.training.valid_size)
    split = splitter.split(data, groups=data['group_id'])
    train_inds, val_inds = next(split)
    train = data.iloc[train_inds]
    valid = data.iloc[val_inds]
    data_module = CreateDataModule(
        train_df=train,
        valid_df=valid,
        #test_df=test,
        batch_size=cfg.training.batch_size,
        max_token_len=cfg.model.max_token_len,
    )
    data_module.setup(stage='fit')

    call_backs = make_callbacks(
        cfg.callbacks.patience_min_delta, cfg.callbacks.patience, checkpoint_path
    )
    model = BertRanker(
        n_classes=cfg.model.n_classes,
        learning_rate=cfg.training.learning_rate,
        pooling_type=cfg.model.pooling_type,
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
                           
    #data_module.setup(stage='test')                       
    #results = trainer.test(ckpt_path=call_backs[1].best_model_path, datamodule=data_module)                      
    #print(results)

if __name__ == "__main__":
    main()
