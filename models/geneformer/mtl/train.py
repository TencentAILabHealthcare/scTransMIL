import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .imports import *
from .model import GeneformerMultiTask
from .utils import calculate_task_specific_metrics, get_layer_freeze_range


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def initialize_wandb(config):
    if config.get("use_wandb", False):
        import wandb

        wandb.init(project=config["wandb_project"], config=config)
        print("Weights & Biases (wandb) initialized and will be used for logging.")
    else:
        print(
            "Weights & Biases (wandb) is not enabled. Logging will use other methods."
        )


def create_model(config, num_labels_list, device):
    model = GeneformerMultiTask(
        config["pretrained_path"],
        num_labels_list,
        dropout_rate=config["dropout_rate"],
        use_task_weights=config["use_task_weights"],
        task_weights=config["task_weights"],
        max_layers_to_freeze=config["max_layers_to_freeze"],
        use_attention_pooling=config["use_attention_pooling"],
    )
    if config["use_data_parallel"]:
        model = nn.DataParallel(model)
    return model.to(device)


def setup_optimizer_and_scheduler(model, config, total_steps):
    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    warmup_steps = int(config["warmup_ratio"] * total_steps)

    if config["lr_scheduler_type"] == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
    elif config["lr_scheduler_type"] == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.5,
        )

    return optimizer, scheduler


def train_epoch(
    model, train_loader, optimizer, scheduler, device, config, writer, epoch
):
    model.train()
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
    for batch_idx, batch in enumerate(progress_bar):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = [
            batch["labels"][task_name].to(device) for task_name in config["task_names"]
        ]

        loss, _, _ = model(input_ids, attention_mask, labels)
        loss.backward()

        if config["gradient_clipping"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])

        optimizer.step()
        scheduler.step()

        writer.add_scalar(
            "Training Loss", loss.item(), epoch * len(train_loader) + batch_idx
        )
        if config.get("use_wandb", False):
            import wandb

            wandb.log({"Training Loss": loss.item()})

        # Update progress bar
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    return loss.item()  # Return the last batch loss


def validate_model(model, val_loader, device, config):
    model.eval()
    val_loss = 0.0
    task_true_labels = {task_name: [] for task_name in config["task_names"]}
    task_pred_labels = {task_name: [] for task_name in config["task_names"]}
    task_pred_probs = {task_name: [] for task_name in config["task_names"]}

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = [
                batch["labels"][task_name].to(device)
                for task_name in config["task_names"]
            ]
            loss, logits, _ = model(input_ids, attention_mask, labels)
            val_loss += loss.item()

            for sample_idx in range(len(batch["input_ids"])):
                for i, task_name in enumerate(config["task_names"]):
                    true_label = batch["labels"][task_name][sample_idx].item()
                    pred_label = torch.argmax(logits[i][sample_idx], dim=-1).item()
                    pred_prob = (
                        torch.softmax(logits[i][sample_idx], dim=-1).cpu().numpy()
                    )
                    task_true_labels[task_name].append(true_label)
                    task_pred_labels[task_name].append(pred_label)
                    task_pred_probs[task_name].append(pred_prob)

    val_loss /= len(val_loader)
    return val_loss, task_true_labels, task_pred_labels, task_pred_probs


def log_metrics(task_metrics, val_loss, config, writer, epochs):
    for task_name, metrics in task_metrics.items():
        print(
            f"{task_name} - Validation F1 Macro: {metrics['f1']:.4f}, Validation Accuracy: {metrics['accuracy']:.4f}"
        )
        if config.get("use_wandb", False):
            import wandb

            wandb.log(
                {
                    f"{task_name} Validation F1 Macro": metrics["f1"],
                    f"{task_name} Validation Accuracy": metrics["accuracy"],
                }
            )

    writer.add_scalar("Validation Loss", val_loss, epochs)
    for task_name, metrics in task_metrics.items():
        writer.add_scalar(f"{task_name} - Validation F1 Macro", metrics["f1"], epochs)
        writer.add_scalar(
            f"{task_name} - Validation Accuracy", metrics["accuracy"], epochs
        )


def save_validation_predictions(
    val_cell_id_mapping,
    task_true_labels,
    task_pred_labels,
    task_pred_probs,
    config,
    trial_number=None,
):
    if trial_number is not None:
        trial_results_dir = os.path.join(config["results_dir"], f"trial_{trial_number}")
        os.makedirs(trial_results_dir, exist_ok=True)
        val_preds_file = os.path.join(trial_results_dir, "val_preds.csv")
    else:
        val_preds_file = os.path.join(config["results_dir"], "manual_run_val_preds.csv")

    rows = []
    for sample_idx in range(len(val_cell_id_mapping)):
        row = {"Cell ID": val_cell_id_mapping[sample_idx]}
        for task_name in config["task_names"]:
            row[f"{task_name} True"] = task_true_labels[task_name][sample_idx]
            row[f"{task_name} Pred"] = task_pred_labels[task_name][sample_idx]
            row[f"{task_name} Probabilities"] = ",".join(
                map(str, task_pred_probs[task_name][sample_idx])
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(val_preds_file, index=False)
    print(f"Validation predictions saved to {val_preds_file}")


def train_model(
    config,
    device,
    train_loader,
    val_loader,
    train_cell_id_mapping,
    val_cell_id_mapping,
    num_labels_list,
):
    set_seed(config["seed"])
    initialize_wandb(config)

    model = create_model(config, num_labels_list, device)
    total_steps = len(train_loader) * config["epochs"]
    optimizer, scheduler = setup_optimizer_and_scheduler(model, config, total_steps)

    log_dir = os.path.join(config["tensorboard_log_dir"], "manual_run")
    writer = SummaryWriter(log_dir=log_dir)

    epoch_progress = tqdm(range(config["epochs"]), desc="Training Progress")
    for epoch in epoch_progress:
        last_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, config, writer, epoch
        )
        epoch_progress.set_postfix({"last_loss": f"{last_loss:.4f}"})

    val_loss, task_true_labels, task_pred_labels, task_pred_probs = validate_model(
        model, val_loader, device, config
    )
    task_metrics = calculate_task_specific_metrics(task_true_labels, task_pred_labels)

    log_metrics(task_metrics, val_loss, config, writer, config["epochs"])
    writer.close()

    save_validation_predictions(
        val_cell_id_mapping, task_true_labels, task_pred_labels, task_pred_probs, config
    )

    if config.get("use_wandb", False):
        import wandb

        wandb.finish()

    print(f"\nFinal Validation Loss: {val_loss:.4f}")
    return val_loss, model  # Return both the validation loss and the trained model


def objective(
    trial,
    train_loader,
    val_loader,
    train_cell_id_mapping,
    val_cell_id_mapping,
    num_labels_list,
    config,
    device,
):
    set_seed(config["seed"])  # Set the seed before each trial
    initialize_wandb(config)

    # Hyperparameters
    config["learning_rate"] = trial.suggest_float(
        "learning_rate",
        config["hyperparameters"]["learning_rate"]["low"],
        config["hyperparameters"]["learning_rate"]["high"],
        log=config["hyperparameters"]["learning_rate"]["log"],
    )
    config["warmup_ratio"] = trial.suggest_float(
        "warmup_ratio",
        config["hyperparameters"]["warmup_ratio"]["low"],
        config["hyperparameters"]["warmup_ratio"]["high"],
    )
    config["weight_decay"] = trial.suggest_float(
        "weight_decay",
        config["hyperparameters"]["weight_decay"]["low"],
        config["hyperparameters"]["weight_decay"]["high"],
    )
    config["dropout_rate"] = trial.suggest_float(
        "dropout_rate",
        config["hyperparameters"]["dropout_rate"]["low"],
        config["hyperparameters"]["dropout_rate"]["high"],
    )
    config["lr_scheduler_type"] = trial.suggest_categorical(
        "lr_scheduler_type", config["hyperparameters"]["lr_scheduler_type"]["choices"]
    )
    config["use_attention_pooling"] = trial.suggest_categorical(
        "use_attention_pooling", [False]
    )

    if config["use_task_weights"]:
        config["task_weights"] = [
            trial.suggest_float(
                f"task_weight_{i}",
                config["hyperparameters"]["task_weights"]["low"],
                config["hyperparameters"]["task_weights"]["high"],
            )
            for i in range(len(num_labels_list))
        ]
        weight_sum = sum(config["task_weights"])
        config["task_weights"] = [
            weight / weight_sum for weight in config["task_weights"]
        ]
    else:
        config["task_weights"] = None

    # Dynamic range for max_layers_to_freeze
    freeze_range = get_layer_freeze_range(config["pretrained_path"])
    config["max_layers_to_freeze"] = trial.suggest_int(
        "max_layers_to_freeze",
        freeze_range["min"],
        freeze_range["max"]
    )

    model = create_model(config, num_labels_list, device)
    total_steps = len(train_loader) * config["epochs"]
    optimizer, scheduler = setup_optimizer_and_scheduler(model, config, total_steps)

    log_dir = os.path.join(config["tensorboard_log_dir"], f"trial_{trial.number}")
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(config["epochs"]):
        train_epoch(
            model, train_loader, optimizer, scheduler, device, config, writer, epoch
        )

    val_loss, task_true_labels, task_pred_labels, task_pred_probs = validate_model(
        model, val_loader, device, config
    )
    task_metrics = calculate_task_specific_metrics(task_true_labels, task_pred_labels)

    log_metrics(task_metrics, val_loss, config, writer, config["epochs"])
    writer.close()

    save_validation_predictions(
        val_cell_id_mapping,
        task_true_labels,
        task_pred_labels,
        task_pred_probs,
        config,
        trial.number,
    )

    trial.set_user_attr("model_state_dict", model.state_dict())
    trial.set_user_attr("task_weights", config["task_weights"])

    trial.report(val_loss, config["epochs"])

    if trial.should_prune():
        raise optuna.TrialPruned()

    if config.get("use_wandb", False):
        import wandb

        wandb.log(
            {
                "trial_number": trial.number,
                "val_loss": val_loss,
                **{
                    f"{task_name}_f1": metrics["f1"]
                    for task_name, metrics in task_metrics.items()
                },
                **{
                    f"{task_name}_accuracy": metrics["accuracy"]
                    for task_name, metrics in task_metrics.items()
                },
                **{
                    k: v
                    for k, v in config.items()
                    if k
                    in [
                        "learning_rate",
                        "warmup_ratio",
                        "weight_decay",
                        "dropout_rate",
                        "lr_scheduler_type",
                        "use_attention_pooling",
                        "max_layers_to_freeze",
                    ]
                },
            }
        )
        wandb.finish()

    return val_loss
