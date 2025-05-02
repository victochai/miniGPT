from datasets import load_dataset, DatasetDict

# This script downloads 10 % of the OpenWebText dataset and splits it into train, validation, and test sets
# It saves the splits as text files in the current directory
# Such approach doesn't work well if you don't have enough RAM, so I eventually I used download_openweb_bin.py

dataset = load_dataset(
    "openwebtext",
    cache_dir="./openwebtext",
    split="train[:10%]", # Load the training split
    )

print(f"\n\nNumber of samples in the dataset: {len(dataset)}")

train_val_test = dataset.train_test_split(test_size=0.1, seed=42)  # 90% train_val, 10% test
train_val = train_val_test['train'].train_test_split(test_size=0.1111, seed=42)

# Final splits
train_dataset = train_val['train']
val_dataset = train_val['test']
test_dataset = train_val_test['test']

print(f"\n\nNumber of samples in the train dataset: {len(train_dataset)}")
print(f"Number of samples in the validation dataset: {len(val_dataset)}")
print(f"Number of samples in the test dataset: {len(test_dataset)}")

def save_text(dataset, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(item["text"].strip() + "\n\n")

save_text(train_dataset, "train.txt")
save_text(val_dataset, "val.txt")
save_text(test_dataset, "test.txt")
