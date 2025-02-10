from datasets import load_dataset

ds = load_dataset("Sp1786/multiclass-sentiment-analysis-dataset")
train_dataset = ds['train']
test_dataset = ds['test']
val_dataset = ds['validation']

# Save to CSV
train_dataset.to_csv("data/train_data.csv", index=True)
val_dataset.to_csv("data/val_data.csv", index=False)
test_dataset.to_csv("data/test_data.csv", index=False)

print("Train data has been saved to train_data.csv")
print("Validation data has been saved to val_data.csv")
print("Test data has been saved to test_data.csv")
