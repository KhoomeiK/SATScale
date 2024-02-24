import os
import argparse
import pickle
from datasets import DatasetDict, load_dataset, Dataset
from huggingface_hub import HfApi, HfFolder


def parse_cnf_file(file_path):
    """Parse a CNF file and return a list of clauses (each clause is a list of integers)."""
    with open(file_path, "r") as file:
        clauses = []
        for line in file:
            # Skip comments and problem line
            if line.startswith("c") or line.startswith("p"):
                continue
            # Convert line to list of integers, excluding the trailing 0
            clause = list(map(int, line.strip().split()))[:-1]
            clauses.append(clause)
        return clauses


def upload_to_huggingface(directory):
    # Get all dataset subdirectories in the specified directory
    dataset_dirs = [
        d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))
    ]

    splits = ("train", "valid", "test", "test_hard")

    for dataset_dir in dataset_dirs:
        dataset_dict = {}
        dataset_name = dataset_dir  # You can customize this name if needed

        # Loop through each split and load the data for the current dataset
        for split in splits:
            split_dir = os.path.join(directory, dataset_dir, split)
            if os.path.exists(split_dir):
                labels = []
                with open(os.path.join(split_dir, "marginals.pkl"), "rb") as f:
                    labels = pickle.load(f) # List[List[float]]

                data = []
                for f in os.listdir(split_dir):
                    if f.endswith(".cnf"):
                        file_num = int(f.split(".")[0])
                        file_path = os.path.join(split_dir, f)
                        clauses = parse_cnf_file(file_path)
                        # Each file's clauses are added as a separate record
                        data.append(
                            {
                                "name": file_num,
                                "clauses": clauses,
                                "label": labels[file_num],
                            }
                        )

                        # simply flip sign on first variable in final clause to generate unsat counterpart
                        unsat_clauses = clauses.copy()
                        unsat_clauses[-1][0] = -unsat_clauses[-1][0]
                        data.append(
                            {
                                "name": f"-{file_num}",
                                "clauses": unsat_clauses,
                                "label": [],
                            }
                        )

                # Directly create a Hugging Face dataset from the list of dictionaries
                dataset_dict[split] = Dataset.from_dict(data)["train"]

        # Create a DatasetDict
        dataset = DatasetDict(dataset_dict)

        # Authenticate with Hugging Face and push to the Hub
        api = HfApi()
        token = HfFolder.get_token()
        if token is None:
            raise ValueError(
                "Hugging Face Hub token not found. Please login using `huggingface-cli login`."
            )
        username = api.whoami(token)["name"]

        # Push each dataset to the Hub under its own name
        dataset.push_to_hub(f"{username}/satscale-{dataset_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload datasets to Huggingface Hub.")
    parser.add_argument(
        "directory", type=str, nargs="?", help="Directory containing the datasets."
    )
    args = parser.parse_args()

    if args.directory:
        upload_to_huggingface(args.directory)
    else:
        print("Please specify the directory containing the datasets.")
