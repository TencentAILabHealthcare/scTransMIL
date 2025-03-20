import os
import shutil

from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from transformers import AutoConfig, BertConfig, BertModel

from .imports import *


def save_model(model, model_save_directory):
    if not os.path.exists(model_save_directory):
        os.makedirs(model_save_directory)

    # Get the state dict
    if isinstance(model, nn.DataParallel):
        model_state_dict = (
            model.module.state_dict()
        )  # Use model.module to access the underlying model
    else:
        model_state_dict = model.state_dict()

    # Remove the "module." prefix from the keys if present
    model_state_dict = {
        k.replace("module.", ""): v for k, v in model_state_dict.items()
    }

    model_save_path = os.path.join(model_save_directory, "pytorch_model.bin")
    torch.save(model_state_dict, model_save_path)

    # Save the model configuration
    if isinstance(model, nn.DataParallel):
        model.module.config.to_json_file(
            os.path.join(model_save_directory, "config.json")
        )
    else:
        model.config.to_json_file(os.path.join(model_save_directory, "config.json"))

    print(f"Model and configuration saved to {model_save_directory}")


def calculate_task_specific_metrics(task_true_labels, task_pred_labels):
    task_metrics = {}
    for task_name in task_true_labels.keys():
        true_labels = task_true_labels[task_name]
        pred_labels = task_pred_labels[task_name]
        f1 = f1_score(true_labels, pred_labels, average="macro")
        accuracy = accuracy_score(true_labels, pred_labels)
        task_metrics[task_name] = {"f1": f1, "accuracy": accuracy}
    return task_metrics


def calculate_combined_f1(combined_labels, combined_preds):
    # Initialize the LabelEncoder
    le = LabelEncoder()

    # Fit and transform combined labels and predictions to numerical values
    le.fit(combined_labels + combined_preds)
    encoded_true_labels = le.transform(combined_labels)
    encoded_pred_labels = le.transform(combined_preds)

    # Print out the mapping for sanity check
    print("\nLabel Encoder Mapping:")
    for index, class_label in enumerate(le.classes_):
        print(f"'{class_label}': {index}")

    # Calculate accuracy
    accuracy = accuracy_score(encoded_true_labels, encoded_pred_labels)

    # Calculate F1 Macro score
    f1 = f1_score(encoded_true_labels, encoded_pred_labels, average="macro")

    return f1, accuracy


# def save_model_without_heads(original_model_save_directory):
#     # Create a new directory for the model without heads
#     new_model_save_directory = original_model_save_directory + "_No_Heads"
#     if not os.path.exists(new_model_save_directory):
#         os.makedirs(new_model_save_directory)

#     # Load the model state dictionary
#     model_state_dict = torch.load(
#         os.path.join(original_model_save_directory, "pytorch_model.bin")
#     )

#     # Initialize a new BERT model without the classification heads
#     config = BertConfig.from_pretrained(
#         os.path.join(original_model_save_directory, "config.json")
#     )
#     model_without_heads = BertModel(config)

#     # Filter the state dict to exclude classification heads
#     model_without_heads_state_dict = {
#         k: v
#         for k, v in model_state_dict.items()
#         if not k.startswith("classification_heads")
#     }

#     # Load the filtered state dict into the model
#     model_without_heads.load_state_dict(model_without_heads_state_dict, strict=False)

#     # Save the model without heads
#     model_save_path = os.path.join(new_model_save_directory, "pytorch_model.bin")
#     torch.save(model_without_heads.state_dict(), model_save_path)

#     # Copy the configuration file
#     shutil.copy(
#         os.path.join(original_model_save_directory, "config.json"),
#         new_model_save_directory,
#     )

#     print(f"Model without classification heads saved to {new_model_save_directory}")


def get_layer_freeze_range(pretrained_path):
    """
    Dynamically determines the number of layers to freeze based on the model depth from its configuration.
    Args:
        pretrained_path (str): Path to the pretrained model directory or model identifier.
    Returns:
        dict: A dictionary with 'min' and 'max' keys indicating the range of layers to freeze.
    """
    if pretrained_path:
        config = AutoConfig.from_pretrained(pretrained_path)
        total_layers = config.num_hidden_layers
        return {"min": 0, "max": total_layers - 1}
    else:
        return {"min": 0, "max": 0}
