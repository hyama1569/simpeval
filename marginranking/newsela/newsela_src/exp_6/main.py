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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler
from transformers import BertModel, BertTokenizer

class SelectedClassBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, can_aug_list):
        self.labels = torch.LongTensor(can_aug_list)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                    for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.dataset = dataset
        self.batch_size = batch_size
    
    def __iter__(self):
        while len(self.labels_set) != 0:
            selected_class = np.random.choice(self.labels_set, 1, replace=False)[0]
            indices = []
            start_ind = self.used_label_indices_count[selected_class]
            end_ind = min(self.used_label_indices_count[selected_class]+self.batch_size, len(self.label_to_indices[selected_class]))
            indices.extend(
                self.label_to_indices[selected_class][start_ind:end_ind]
            )
            if self.used_label_indices_count[selected_class] + self.batch_size > len(self.label_to_indices[selected_class]):
                self.labels_set.remove(selected_class)
            self.used_label_indices_count[selected_class] += self.batch_size
            yield indices

    def __len__(self):
        return len(self.dataset) // self.batch_size

class AugmentedDataset(Dataset):
    def __init__(
        self, 
        data: pd.DataFrame, 
        tokenizer: BertTokenizer, 
        max_token_len: int,
        orig_column_name: str,
        simp_column_name: str,
        inter_column_name: str,
        label_column_name: str,
        case_num_column_name: str,
        can_aug_column_name: str,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.orig_column_name = orig_column_name
        self.simp_column_name = simp_column_name
        self.inter_column_name = inter_column_name
        self.label_column_name = label_column_name
        self.case_num_column_name = case_num_column_name
        self.can_aug_column_name = can_aug_column_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        orig = data_row[self.orig_column_name]
        simp = data_row[self.simp_column_name]
        inter = data_row[self.inter_column_name]
        label = data_row[self.label_column_name]
        case_num = data_row[self.case_num_column_name]
        can_aug = data_row[self.can_aug_column_name]

        if can_aug == 0:
            enc_orig_orig = self.tokenizer.encode_plus(orig, orig, add_special_tokens=True, max_length=self.max_token_len, padding="max_length", truncation=True, return_attention_mask=True, return_tensors='pt')
            enc_orig_simp = self.tokenizer.encode_plus(orig, simp, add_special_tokens=True, max_length=self.max_token_len, padding="max_length", truncation=True, return_attention_mask=True, return_tensors='pt')
            enc_simp_simp = self.tokenizer.encode_plus(simp, simp, add_special_tokens=True, max_length=self.max_token_len, padding="max_length", truncation=True, return_attention_mask=True, return_tensors='pt')
            return dict(
                orig_orig=dict(
                    input_ids=enc_orig_orig["input_ids"].flatten(),
                    attention_mask=enc_orig_orig["attention_mask"].flatten(),
                ),
                orig_simp=dict(
                    input_ids=enc_orig_simp["input_ids"].flatten(),
                    attention_mask=enc_orig_simp["attention_mask"].flatten(),
                ),
                simp_simp=dict(
                    input_ids=enc_simp_simp["input_ids"].flatten(),
                    attention_mask=enc_simp_simp["attention_mask"].flatten(),
                ),
                labels=torch.tensor(label),
                case_nums=torch.tensor(case_num),
                can_aug=can_aug,
            )
        elif can_aug == 1:
            enc_orig_orig = self.tokenizer.encode_plus(orig, orig, add_special_tokens=True, max_length=self.max_token_len, padding="max_length", truncation=True, return_attention_mask=True, return_tensors='pt')
            enc_orig_simp = self.tokenizer.encode_plus(orig, simp, add_special_tokens=True, max_length=self.max_token_len, padding="max_length", truncation=True, return_attention_mask=True, return_tensors='pt')
            enc_simp_simp = self.tokenizer.encode_plus(simp, simp, add_special_tokens=True, max_length=self.max_token_len, padding="max_length", truncation=True, return_attention_mask=True, return_tensors='pt')
            enc_orig_inter = self.tokenizer.encode_plus(orig, inter, add_special_tokens=True, max_length=self.max_token_len, padding="max_length", truncation=True, return_attention_mask=True, return_tensors='pt')
            enc_inter_simp = self.tokenizer.encode_plus(inter, simp, add_special_tokens=True, max_length=self.max_token_len, padding="max_length", truncation=True, return_attention_mask=True, return_tensors='pt')
            enc_inter_inter = self.tokenizer.encode_plus(inter, inter, add_special_tokens=True, max_length=self.max_token_len, padding="max_length", truncation=True, return_attention_mask=True, return_tensors='pt')
            return dict(
                orig_orig=dict(
                    input_ids=enc_orig_orig["input_ids"].flatten(),
                    attention_mask=enc_orig_orig["attention_mask"].flatten(),
                ),
                orig_simp=dict(
                    input_ids=enc_orig_simp["input_ids"].flatten(),
                    attention_mask=enc_orig_simp["attention_mask"].flatten(),
                ),
                simp_simp=dict(
                    input_ids=enc_simp_simp["input_ids"].flatten(),
                    attention_mask=enc_simp_simp["attention_mask"].flatten(),
                ),
                orig_inter=dict(
                    input_ids=enc_orig_inter["input_ids"].flatten(),
                    attention_mask=enc_orig_inter["attention_mask"].flatten(),
                ),
                inter_simp=dict(
                    input_ids=enc_inter_simp["input_ids"].flatten(),
                    attention_mask=enc_inter_simp["attention_mask"].flatten(),
                ),
                inter_inter=dict(
                    input_ids=enc_inter_inter["input_ids"].flatten(),
                    attention_mask=enc_inter_inter["attention_mask"].flatten(),
                ),
                labels=torch.tensor(label),
                case_nums=torch.tensor(case_num),
                can_aug=can_aug,
            )

class CreateDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame = None, 
        valid_df: pd.DataFrame = None, 
        test_df: pd.DataFrame = None, 
        batch_size: int = None, 
        max_token_len: int = None, 
        orig_column_name: str = 'original',
        simp_column_name: str = 'simple',
        inter_column_name: str = 'inter',
        label_column_name: str = 'label',
        case_num_column_name: str = 'case_num',
        can_aug_column_name: str = 'can_aug',
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
        self.inter_column_name = inter_column_name
        self.label_column_name = label_column_name
        self.case_num_column_name = case_num_column_name
        self.can_aug_column_name = can_aug_column_name
        self.can_aug_list_train = self.train_df[self.can_aug_column_name].tolist()
        self.can_aug_list_valid = self.valid_df[self.can_aug_column_name].tolist()
        self.can_aug_list_test = self.test_df[self.can_aug_column_name].tolist()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)

    def setup(self, stage):
        if stage == "fit":
          self.train_dataset = AugmentedDataset(
              self.train_df, 
              self.tokenizer, 
              self.max_token_len,
              self.orig_column_name,
              self.simp_column_name,
              self.inter_column_name,
              self.label_column_name,
              self.case_num_column_name,
              self.can_aug_column_name,
            )
          self.vaild_dataset = AugmentedDataset(
              self.valid_df, 
              self.tokenizer, 
              self.max_token_len,
              self.orig_column_name,
              self.simp_column_name,
              self.inter_column_name,
              self.label_column_name,
              self.case_num_column_name,
              self.can_aug_column_name,
            )
        if stage == "test":
          self.test_dataset = AugmentedDataset(
              self.test_df, 
              self.tokenizer, 
              self.max_token_len,
              self.orig_column_name,
              self.simp_column_name,
              self.inter_column_name,
              self.label_column_name,
              self.case_num_column_name,
              self.can_aug_column_name,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=os.cpu_count(), batch_sampler=SelectedClassBatchSampler(self.train_dataset, self.batch_size, can_aug_list=self.can_aug_list_train))

    def val_dataloader(self):
        return DataLoader(self.vaild_dataset, num_workers=os.cpu_count(), batch_sampler=SelectedClassBatchSampler(self.train_dataset, self.batch_size, can_aug_list=self.can_aug_list_valid))

    def test_dataloader(self):
        return DataLoader(self.test_dataset, num_workers=os.cpu_count(), batch_sampler=SelectedClassBatchSampler(self.train_dataset, self.batch_size, can_aug_list=self.can_aug_list_test))

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
        self.criterion = nn.MarginRankingLoss(margin=1.0)
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
        preds = torch.flatten(preds)
        return preds, output

    def training_step(self, batch, batch_idx):
        #if selected batch with can_aug == 1
        if batch["can_aug"].any():
            orig_orig_preds, _ = self.forward(input_ids=batch["orig_orig"]["input_ids"], attention_mask=batch["orig_orig"]["attention_mask"])
            orig_simp_preds, _ = self.forward(input_ids=batch["orig_simp"]["input_ids"], attention_mask=batch["orig_simp"]["attention_mask"])
            simp_simp_preds, _ = self.forward(input_ids=batch["simp_simp"]["input_ids"], attention_mask=batch["simp_simp"]["attention_mask"])
            orig_inter_preds, _ = self.forward(input_ids=batch["orig_inter"]["input_ids"], attention_mask=batch["orig_inter"]["attention_mask"])
            inter_simp_preds, _ = self.forward(input_ids=batch["inter_simp"]["input_ids"], attention_mask=batch["inter_simp"]["attention_mask"])
            inter_inter_preds, _ = self.forward(input_ids=batch["inter_inter"]["input_ids"], attention_mask=batch["inter_inter"]["attention_mask"])

            loss = 0
            # w/o aug margin loss
            loss += self.criterion(orig_simp_preds, orig_orig_preds, batch["labels"])
            loss += self.criterion(orig_simp_preds, simp_simp_preds, batch["labels"])
            # w/ aug margin loss
            loss += self.criterion(orig_inter_preds, orig_orig_preds, batch["labels"])
            loss += self.criterion(orig_inter_preds, inter_inter_preds, batch["labels"])
            loss += self.criterion(orig_inter_preds, simp_simp_preds, batch["labels"])
            loss += self.criterion(inter_simp_preds, orig_orig_preds, batch["labels"])
            loss += self.criterion(inter_simp_preds, inter_inter_preds, batch["labels"])
            loss += self.criterion(inter_simp_preds, simp_simp_preds, batch["labels"])
            loss += self.criterion(orig_simp_preds, orig_inter_preds, batch["labels"])
            loss += self.criterion(orig_simp_preds, inter_simp_preds, batch["labels"])
            loss /= 10

            return {'loss': loss, 
                    'batch_preds': [orig_orig_preds, orig_simp_preds, simp_simp_preds, orig_inter_preds, inter_simp_preds, inter_inter_preds],
                    'batch_labels': batch["labels"],
                    'batch_can_aug': batch["can_aug"],
                    }
        else:
            orig_orig_preds, _ = self.forward(input_ids=batch["orig_orig"]["input_ids"], attention_mask=batch["orig_orig"]["attention_mask"])
            orig_simp_preds, _ = self.forward(input_ids=batch["orig_simp"]["input_ids"], attention_mask=batch["orig_simp"]["attention_mask"])
            simp_simp_preds, _ = self.forward(input_ids=batch["simp_simp"]["input_ids"], attention_mask=batch["simp_simp"]["attention_mask"])

            loss = 0
            # w/o aug margin loss
            loss += self.criterion(orig_simp_preds, orig_orig_preds, batch["labels"])
            loss += self.criterion(orig_simp_preds, simp_simp_preds, batch["labels"])
            loss /= 2

            return {'loss': loss, 
                    'batch_preds': [orig_orig_preds, orig_simp_preds, simp_simp_preds],
                    'batch_labels': batch["labels"],
                    'batch_can_aug': batch["can_aug"],
                    }
    
    def validation_step(self, batch, batch_idx):
        #if selected batch with can_aug == 1
        if batch["can_aug"].any():
            orig_orig_preds, _ = self.forward(input_ids=batch["orig_orig"]["input_ids"], attention_mask=batch["orig_orig"]["attention_mask"])
            orig_simp_preds, _ = self.forward(input_ids=batch["orig_simp"]["input_ids"], attention_mask=batch["orig_simp"]["attention_mask"])
            simp_simp_preds, _ = self.forward(input_ids=batch["simp_simp"]["input_ids"], attention_mask=batch["simp_simp"]["attention_mask"])
            orig_inter_preds, _ = self.forward(input_ids=batch["orig_inter"]["input_ids"], attention_mask=batch["orig_inter"]["attention_mask"])
            inter_simp_preds, _ = self.forward(input_ids=batch["inter_simp"]["input_ids"], attention_mask=batch["inter_simp"]["attention_mask"])
            inter_inter_preds, _ = self.forward(input_ids=batch["inter_inter"]["input_ids"], attention_mask=batch["inter_inter"]["attention_mask"])

            loss = 0
            # w/o aug margin loss
            loss += self.criterion(orig_simp_preds, orig_orig_preds, batch["labels"])
            loss += self.criterion(orig_simp_preds, simp_simp_preds, batch["labels"])
            # w/ aug margin loss
            loss += self.criterion(orig_inter_preds, orig_orig_preds, batch["labels"])
            loss += self.criterion(orig_inter_preds, inter_inter_preds, batch["labels"])
            loss += self.criterion(orig_inter_preds, simp_simp_preds, batch["labels"])
            loss += self.criterion(inter_simp_preds, orig_orig_preds, batch["labels"])
            loss += self.criterion(inter_simp_preds, inter_inter_preds, batch["labels"])
            loss += self.criterion(inter_simp_preds, simp_simp_preds, batch["labels"])
            loss += self.criterion(orig_simp_preds, orig_inter_preds, batch["labels"])
            loss += self.criterion(orig_simp_preds, inter_simp_preds, batch["labels"])
            loss /= 10

            return {'loss': loss, 
                    'batch_preds': [orig_orig_preds, orig_simp_preds, simp_simp_preds, orig_inter_preds, inter_simp_preds, inter_inter_preds],
                    'batch_labels': batch["labels"],
                    'batch_can_aug': batch["can_aug"],
                    }
        else:
            orig_orig_preds, _ = self.forward(input_ids=batch["orig_orig"]["input_ids"], attention_mask=batch["orig_orig"]["attention_mask"])
            orig_simp_preds, _ = self.forward(input_ids=batch["orig_simp"]["input_ids"], attention_mask=batch["orig_simp"]["attention_mask"])
            simp_simp_preds, _ = self.forward(input_ids=batch["simp_simp"]["input_ids"], attention_mask=batch["simp_simp"]["attention_mask"])

            loss = 0
            # w/o aug margin loss
            loss += self.criterion(orig_simp_preds, orig_orig_preds, batch["labels"])
            loss += self.criterion(orig_simp_preds, simp_simp_preds, batch["labels"])
            loss /= 2

            return {'loss': loss, 
                    'batch_preds': [orig_orig_preds, orig_simp_preds, simp_simp_preds],
                    'batch_labels': batch["labels"],
                    'batch_can_aug': batch["can_aug"],
                    }

    def test_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)
    
    def training_epoch_end(self, outputs, mode="train"):
        epoch_loss = torch.stack([x["loss"] for x in outputs]).mean().item()
        self.log(f"{mode}_loss", epoch_loss, logger=True)

    def validation_epoch_end(self, outputs, mode="val"):
        epoch_loss = torch.stack([x["loss"] for x in outputs]).mean().item()
        self.log(f"{mode}_loss", epoch_loss, logger=True)

    def test_epoch_end(self, outputs):
        return self.training_epoch_end(outputs, "test")

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
    data['label'] = 1
    data.rename(columns={'comp':'original', 'simp':'simple'}, inplace=True)
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
                           
    data_module.setup(stage='test')                       
    results = trainer.test(ckpt_path=call_backs[1].best_model_path, datamodule=data_module)                      
    print(results)

if __name__ == "__main__":
    main()
