import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split


def read_cnf_to_vectors(file_path, dim=256):
    vectors = []
    max_index = 0
    total_clauses = 0

    with open(file_path, "r") as file:
        for line in file:
            if line.startswith("p"):
                parts = line.split()
                max_index = dim  # int(parts[2])
                total_clauses = int(parts[3])
            elif not line[0] in ("c", "%", "0") and line.strip():
                vector = [0] * max_index
                numbers = [int(x) for x in line.split() if x != "0"]
                for num in numbers:
                    if num < 0:
                        vector[abs(num) - 1] = -1
                    else:
                        vector[num - 1] = 1
                vectors.append(vector)

    assert len(vectors) == total_clauses

    return vectors, max_index, total_clauses


class CNFDataset(Dataset):
    def __init__(self, file_paths):
        self.data = []
        self.input_dim = 0
        self.max_seq_len = 0

        for file_path in file_paths:
            label = 1 if file_path.split("/")[-1].startswith("uf") else 0

            vectors, input_dim, seq_len = read_cnf_to_vectors(file_path)
            vectors_tensor = torch.tensor(vectors, dtype=torch.float32)
            # vectors_tensor = vectors_tensor.unsqueeze(1)

            self.data.append((vectors_tensor, label))
            self.input_dim = max(self.input_dim, input_dim)
            self.max_seq_len = max(self.max_seq_len, seq_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    seqs, labels = zip(*batch)
    padded_seqs = nn.utils.rnn.pad_sequence(seqs, batch_first=False, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.float32)
    return padded_seqs, labels


def train_eval_test_split(dataset, train_size=0.8, test_size=0.1):
    indices = list(range(len(dataset)))
    train_indices, temp_indices = train_test_split(indices, train_size=train_size)
    val_size = test_size / (1 - train_size)
    eval_indices, test_indices = train_test_split(temp_indices, test_size=val_size)

    train_dataset = Subset(dataset, train_indices)
    eval_dataset = Subset(dataset, eval_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, eval_dataset, test_dataset
