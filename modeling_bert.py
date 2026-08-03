import torch
import torch.nn as nn 
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, BertModel
from torch.utils.data import DataLoader  , Dataset 
import wandb
from tqdm import tqdm 


class CustomDataset(Dataset):
    def __init__(self , file_name="./Train.csv" , tokenizer_name="google-bert/bert-base-cased" , max_length=128):
        self.df = pd.read_csv(file_name,encoding="latin1")
        self.df = self.df[["tweet","type"]]
        self.text = self.df["tweet"].astype(str).tolist()
        self.label = self.df["type"].astype(str).tolist()
        self.max_length=max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        self.label_id ={
            "sexual_violence":0 ,
            "Physical_violence":1,
            "emotional_violence":2,
            "economic_violence":3,
            "Harmful_Traditional_practice":4,
        }
        
    def __len__(self):
        return len(self.df) 
    def __getitem__(self, idx):
        text = self.text[idx]
        label = self.label[idx]
        
        ids = self.tokenizer(
            text,
            padding="max_length" , 
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        labels = self.label_id[label]
        
        return {
            "input_ids":ids["input_ids"].squeeze(0) , 
            "label":torch.tensor(labels , dtype=torch.long)
        }
        
        
class Model(nn.Module):
    def __init__(self,model_name="google-bert/bert-base-cased" , num_classes=5):
        super().__init__()
        self.model = BertModel.from_pretrained(model_name) 
        self.hidden_dim = self.model.config.hidden_size
        self.proj_lin = nn.Linear(self.hidden_dim , num_classes)
    
    def forward(self , input_ids):
        x = self.model(input_ids) 
        x= x.last_hidden_state[:,0]
        x = self.proj_lin(x)
        
        return x
    
    
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
 
 
    for batch in tqdm(dataloader  , total=len(dataloader)):
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)
 
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
 
        total_loss += loss.item() * input_ids.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
 
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
 
    for batch in tqdm(dataloader , total=len(dataloader)):
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)
 
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
 
        total_loss += loss.item() * input_ids.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
 
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    file_name = "./Train_sample.csv"
    model_name = "google-bert/bert-base-cased"
    batch_size = 4
    num_epochs = 1
    lr = 2e-5
    val_split = 0.15
    num_classes = 5
    
    config = {
        "model_name": model_name,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "lr": lr,
        "val_split": val_split,
    }
 
    wandb.init(
        project="bert-violence-classification",
        config=config,
    )
    
 
    dataset = CustomDataset(file_name=file_name, tokenizer_name=model_name, max_length=64)
 
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
 
    model = Model(model_name=model_name, num_classes=num_classes).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
 
    # le dataset est tres desequilibre (32k vs 188 exemples selon les classes)
    # donc on pondere la loss pour ne pas ignorer les classes minoritaires
    counts = pd.Series(dataset.label).value_counts()
    counts = counts.reindex(dataset.label_id.keys())
    class_weights = torch.tensor(counts.sum() / (num_classes * counts), dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    
    best_val_acc=0


    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
 
        print(f"Epoch {epoch + 1}/{num_epochs} "
              f"| Train loss: {train_loss:.4f}, acc: {train_acc:.4f} "
              f"| Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")
 
 
        wandb.log({
            "epoch": epoch + 1,
            "train/epoch_loss": train_loss,
            "train/epoch_accuracy": train_acc,
            "val/epoch_loss": val_loss,
            "val/epoch_accuracy": val_acc,
        })
 
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wandb.run.summary["best_val_accuracy"] = best_val_acc

    torch.save(model.state_dict(), "violence_model.pt")
    
    wandb.finish()


if __name__=="__main__":
    main()
