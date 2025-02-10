
from transformers import AutoTokenizer
from data.preprocess import test_dataset, train_dataset, val_dataset
from torch.utils.data import Dataset
import pandas as pd
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})



class TweetDataset(Dataset):
    def __init__(self, dataframe, tokenizer_name="gpt2", max_length=128):
        self.data =  pd.read_csv(dataframe)

        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        # Fix missing pad_token for models like GPT-2
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Ensure padding side matches model requirements (left for GPT-2)
        self.tokenizer.padding_side = "left"  # Critical for attention masks!

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text, label = row['text'], row['label']

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',  # Returns batch-first tensors
            return_attention_mask=True,
        )

        # Remove batch dimension added by return_tensors='pt'
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding['label'] = torch.tensor(label, dtype=torch.long)
        
        return encoding

train_dataset = TweetDataset("data/train_data.csv")
test_dataset = TweetDataset("data/test_data.csv")
val_dataset = TweetDataset("data/val_data.csv")
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer)
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=data_collator,
    num_workers=10,  # Adjust based on your system's capabilities
    pin_memory=True  # Useful if you're using a GPU
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=data_collator,
    num_workers=10,  # Adjust based on your system's capabilities
    pin_memory=True  # Useful if you're using a GPU
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=data_collator,
    num_workers=10  # Adjust based on your system's capabilities
)
print(f"{len(train_loader)} training batches")
print(f"{len(test_loader)} test batches")
print(f"{len(val_loader)} validation batches")
