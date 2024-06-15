import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def collect_files(directory, file_extension):
    file_contents = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(file_extension):
                try:
                    content = read_file(os.path.join(root, file))
                    file_contents.append(content)
                except UnicodeDecodeError as e:
                    print(f"Error reading {file}: {e}")
    return file_contents

odoo_files = collect_files('/opt/odoo16', '.py')

all_files = odoo_files

def train_tokenizer(file_contents):
    tokenizer = Tokenizer(models.WordPiece())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordPieceTrainer(vocab_size=30522, min_frequency=2)
    tokenizer.train_from_iterator(file_contents, trainer)
    tokenizer.save("tokenizer.json")

train_tokenizer(all_files)
print("completed")
