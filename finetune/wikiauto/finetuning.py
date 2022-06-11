import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.optim as optim
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
ORIG_COLUMN = "original"
SIMP_COLUMN = "simple"
LABEL_COLUMN = "label"
pl.seed_everything(0, workers=True)

class CreateDataset(Dataset):
    def __init__(self, data, tokenizer, max_token_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        origs = data_row[ORIG_COLUMN]
        simps = data_row[SIMP_COLUMN]
        labels = data_row[LABEL_COLUMN]

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
    def __init__(self, train_df=None, valid_df=None, test_df=None, batch_size=16, max_token_len=512, pretrained_model='bert-base-uncased'):
        super().__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)

    def setup(self, stage):
        if stage == "fit" or stage is None:
          self.train_dataset = CreateDataset(self.train_df, self.tokenizer, self.max_token_len)
          self.vaild_dataset = CreateDataset(self.valid_df, self.tokenizer, self.max_token_len)
        if stage == "test" or stage is None:
          self.test_dataset = CreateDataset(self.test_df, self.tokenizer, self.max_token_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count())

    def val_dataloader(self):
        return DataLoader(self.vaild_dataset, batch_size=self.batch_size, num_workers=os.cpu_count())

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=os.cpu_count())

class BertClassifier(pl.LightningModule):
    def __init__(self, n_classes: int, n_epochs=None, pretrained_model='bert-base-uncased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model, output_hidden_states=True, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.n_epochs = n_epochs
        self.criterion = nn.CrossEntropyLoss()

        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        preds = self.classifier(output.pooler_output)
        loss = 0
        if labels is not None:
            loss = self.criterion(preds, labels)
        return loss, preds, output
      
    def predict_step(self, batch, batch_idx):
        loss, preds, output = self.forward(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"],
                                    labels=batch["labels"])
        return preds, output

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

    def validation_epoch_end(self, outputs, mode="val"):
        epoch_preds = torch.cat([x['batch_preds'] for x in outputs])
        epoch_labels = torch.cat([x['batch_labels'] for x in outputs])
        epoch_loss = self.criterion(epoch_preds, epoch_labels)
        self.log(f"{mode}_loss", epoch_loss, logger=True)

        num_correct = (epoch_preds.argmax(dim=1) == epoch_labels).sum().item()
        epoch_accuracy = num_correct / len(epoch_labels)
        self.log(f"{mode}_accuracy", epoch_accuracy, logger=True)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, "test")

    def configure_optimizers(self):
        optimizer = optim.Adam([
            {'params': self.bert.encoder.layer[-1].parameters(), 'lr': 5e-5},
            {'params': self.classifier.parameters(), 'lr': 1e-4}
        ])

        return [optimizer]

if __name__ == '__main__':
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.05,
        patience=3,
        mode='min'
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath="./src/checkpoints",
        filename='{epoch}',
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    N_EPOCHS = 10
    data = pd.read_pickle('./src/wikiauto_dataframe.pickle')
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=0)
    train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=0)
    data_module = CreateDataModule(train_df=train_df, valid_df=valid_df, test_df=test_df)
    data_module.setup(stage=None)

    model = BertClassifier(n_classes=2, n_epochs=N_EPOCHS)
    trainer = pl.Trainer(max_epochs=N_EPOCHS, gpus=1, accelerator='gpu', strategy='dp', callbacks=[checkpoint_callback, early_stop_callback])
    trainer.fit(model, data_module)
    result = trainer.test(ckpt_path=checkpoint_callback.best_model_path, datamodule=data_module)
