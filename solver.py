from functools import partial
import glob
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import wandb

from data import CNFDataset, train_eval_test_split, collate_fn

class TransformerBinaryClassifier(nn.Module):
    def __init__(self, input_dim, num_layers, num_heads):
        super(TransformerBinaryClassifier, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, src):
        output = self.transformer_encoder(src)
        output = torch.sum(output, dim=0)
        output = self.fc(output)
        return torch.sigmoid(output)


def train_model(train_dataset, eval_dataset):
    with wandb.init():
        hparams = wandb.config
        model = TransformerBinaryClassifier(
            input_dim=hparams["input_dim"],
            num_layers=hparams["num_layers"],
            num_heads=hparams["num_heads"],
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        train_loader = DataLoader(
            train_dataset,
            batch_size=hparams["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=hparams["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
        )

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=hparams["learning_rate"])
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=hparams["step_size"], gamma=hparams["gamma"]
        )

        for epoch in range(hparams["num_epochs"]):
            model.train()
            total_loss = 0
            for seq, labels in train_loader:
                seq, labels = seq.to(device), labels.to(device)

                outputs = model(seq).squeeze(dim=1)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            scheduler.step()

            avg_loss = total_loss / len(train_loader)
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                }
            )

            model.eval()
            with torch.no_grad():
                eval_loss = 0
                correct_predictions = 0
                total_predictions = 0
                for seq, labels in eval_loader:
                    seq, labels = seq.to(device), labels.to(device)
                    outputs = model(seq).squeeze(dim=1)
                    loss = criterion(outputs, labels)
                    eval_loss += loss.item()

                    # Calculate accuracy
                    predicted_labels = outputs > 0.5
                    correct_predictions += (
                        (predicted_labels == labels.bool()).sum().item()
                    )
                    total_predictions += labels.size(0)

                avg_eval_loss = eval_loss / len(eval_loader)
                accuracy = correct_predictions / total_predictions
                wandb.log(
                    {"epoch": epoch, "eval_loss": avg_eval_loss, "accuracy": accuracy}
                )

            print(
                f"Epoch [{epoch+1}/{hparams['num_epochs']}], Train Loss: {avg_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}, Accuracy: {accuracy:.4f}"
            )


if __name__ == "__main__":
    cnf_files = glob.glob("data/*.cnf")
    cnf_dataset = CNFDataset(cnf_files)
    train_dataset, eval_dataset, test_dataset = train_eval_test_split(cnf_dataset)

    # hparams = {
    #     "batch_size": 32,
    #     "num_epochs": 20,
    #     "learning_rate": 0.001,
    #     "num_layers": 3,
    #     "num_heads": 1,
    # "input_dim": cnf_dataset.input_dim,
    # "max_seq_len": cnf_dataset.max_seq_len,
    # }

    # wandb.init(
    #     project="SATScale",
    #     entity="rspandey",
    #     config=hparams,
    # )
    # train_model(train_dataset, eval_dataset)

    sweep_config = {
        "method": "bayes",
        "metric": {"name": "eval_loss", "goal": "minimize"},
        "parameters": {
            "learning_rate": {"min": 0.0001, "max": 0.01},
            "num_layers": {"values": [1, 2, 4, 8, 16]},
            "num_heads": {"values": [1, 2, 4, 8]},
            "batch_size": {"values": [16, 32, 64]},
            "num_epochs": {"values": [5, 20, 50]},
            "step_size": {"values": [5, 10, 20]},
            "gamma": {"values": [0.1, 0.5, 0.9]},
            "input_dim": {"values": [cnf_dataset.input_dim]},
            "max_seq_len": {"values": [cnf_dataset.max_seq_len]},
        },
    }
    sweep_id = wandb.sweep(sweep_config, project="SATScale", entity="rspandey")
    train_model_partial = partial(
        train_model, train_dataset=train_dataset, eval_dataset=eval_dataset
    )
    wandb.agent(sweep_id, train_model_partial)
