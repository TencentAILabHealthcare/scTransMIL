import os

from .collators import DataCollatorForMultitaskCellClassification
from .imports import *


def load_and_preprocess_data(dataset_path, config, is_test=False, dataset_type=""):
    try:
        dataset = load_from_disk(dataset_path)

        task_names = [f"task{i+1}" for i in range(len(config["task_columns"]))]
        task_to_column = dict(zip(task_names, config["task_columns"]))
        config["task_names"] = task_names

        if not is_test:
            available_columns = set(dataset.column_names)
            for column in task_to_column.values():
                if column not in available_columns:
                    raise KeyError(
                        f"Column {column} not found in the dataset. Available columns: {list(available_columns)}"
                    )

        label_mappings = {}
        task_label_mappings = {}
        cell_id_mapping = {}
        num_labels_list = []

        # Load or create task label mappings
        if not is_test:
            for task, column in task_to_column.items():
                unique_values = sorted(set(dataset[column]))  # Ensure consistency
                label_mappings[column] = {
                    label: idx for idx, label in enumerate(unique_values)
                }
                task_label_mappings[task] = label_mappings[column]
                num_labels_list.append(len(unique_values))

            # Print the mappings for each task with dataset type prefix
            for task, mapping in task_label_mappings.items():
                print(
                    f"{dataset_type.capitalize()} mapping for {task}: {mapping}"
                )  # sanity check, for train/validation splits

            # Save the task label mappings as a pickle file
            with open(f"{config['results_dir']}/task_label_mappings.pkl", "wb") as f:
                pickle.dump(task_label_mappings, f)
        else:
            # Load task label mappings from pickle file for test data
            with open(f"{config['results_dir']}/task_label_mappings.pkl", "rb") as f:
                task_label_mappings = pickle.load(f)

            # Infer num_labels_list from task_label_mappings
            for task, mapping in task_label_mappings.items():
                num_labels_list.append(len(mapping))

        # Store unique cell IDs in a separate dictionary
        for idx, record in enumerate(dataset):
            cell_id = record.get("unique_cell_id", idx)
            cell_id_mapping[idx] = cell_id

        # Transform records to the desired format
        transformed_dataset = []
        for idx, record in enumerate(dataset):
            transformed_record = {}
            transformed_record["input_ids"] = torch.tensor(
                record["input_ids"], dtype=torch.long
            )

            # Use index-based cell ID for internal tracking
            transformed_record["cell_id"] = idx

            if not is_test:
                # Prepare labels
                label_dict = {}
                for task, column in task_to_column.items():
                    label_value = record[column]
                    label_index = task_label_mappings[task][label_value]
                    label_dict[task] = label_index
                transformed_record["label"] = label_dict
            else:
                # Create dummy labels for test data
                label_dict = {task: -1 for task in config["task_names"]}
                transformed_record["label"] = label_dict

            transformed_dataset.append(transformed_record)

        return transformed_dataset, cell_id_mapping, num_labels_list
    except KeyError as e:
        print(f"Missing configuration or dataset key: {e}")
    except Exception as e:
        print(f"An error occurred while loading or preprocessing data: {e}")
        return None, None, None


def preload_and_process_data(config):
    # Load and preprocess data once
    train_dataset, train_cell_id_mapping, num_labels_list = load_and_preprocess_data(
        config["train_path"], config, dataset_type="train"
    )
    val_dataset, val_cell_id_mapping, _ = load_and_preprocess_data(
        config["val_path"], config, dataset_type="validation"
    )
    return (
        train_dataset,
        train_cell_id_mapping,
        val_dataset,
        val_cell_id_mapping,
        num_labels_list,
    )


def get_data_loader(preprocessed_dataset, batch_size):
    nproc = os.cpu_count()  ### I/O operations

    data_collator = DataCollatorForMultitaskCellClassification()

    loader = DataLoader(
        preprocessed_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=nproc,
        pin_memory=True,
    )
    return loader


def preload_data(config):
    # Preprocessing the data before the Optuna trials start
    train_loader = get_data_loader("train", config)
    val_loader = get_data_loader("val", config)
    return train_loader, val_loader


def load_and_preprocess_test_data(config):
    """
    Load and preprocess test data, treating it as unlabeled.
    """
    return load_and_preprocess_data(config["test_path"], config, is_test=True)


def prepare_test_loader(config):
    """
    Prepare DataLoader for the test dataset.
    """
    test_dataset, cell_id_mapping, num_labels_list = load_and_preprocess_test_data(
        config
    )
    test_loader = get_data_loader(test_dataset, config["batch_size"])
    return test_loader, cell_id_mapping, num_labels_list
