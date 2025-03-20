import random

from .data import get_data_loader, preload_and_process_data
from .imports import *
from .model import GeneformerMultiTask
from .train import objective, train_model
from .utils import save_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_manual_tuning(config):
    # Set seed for reproducibility
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        train_dataset,
        train_cell_id_mapping,
        val_dataset,
        val_cell_id_mapping,
        num_labels_list,
    ) = preload_and_process_data(config)
    train_loader = get_data_loader(train_dataset, config["batch_size"])
    val_loader = get_data_loader(val_dataset, config["batch_size"])

    # Print the manual hyperparameters being used
    print("\nManual hyperparameters being used:")
    for key, value in config["manual_hyperparameters"].items():
        print(f"{key}: {value}")
    print()  # Add an empty line for better readability

    # Use the manual hyperparameters
    for key, value in config["manual_hyperparameters"].items():
        config[key] = value

    # Train the model
    val_loss, trained_model = train_model(
        config,
        device,
        train_loader,
        val_loader,
        train_cell_id_mapping,
        val_cell_id_mapping,
        num_labels_list,
    )

    print(f"\nValidation loss with manual hyperparameters: {val_loss}")

    # Save the trained model
    model_save_directory = os.path.join(
        config["model_save_path"], "GeneformerMultiTask"
    )
    save_model(trained_model, model_save_directory)

    # Save the hyperparameters
    hyperparams_to_save = {
        **config["manual_hyperparameters"],
        "dropout_rate": config["dropout_rate"],
        "use_task_weights": config["use_task_weights"],
        "task_weights": config["task_weights"],
        "max_layers_to_freeze": config["max_layers_to_freeze"],
        "use_attention_pooling": config["use_attention_pooling"],
    }
    hyperparams_path = os.path.join(model_save_directory, "hyperparameters.json")
    with open(hyperparams_path, "w") as f:
        json.dump(hyperparams_to_save, f)
    print(f"Manual hyperparameters saved to {hyperparams_path}")

    return val_loss


def run_optuna_study(config):
    # Set seed for reproducibility
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        train_dataset,
        train_cell_id_mapping,
        val_dataset,
        val_cell_id_mapping,
        num_labels_list,
    ) = preload_and_process_data(config)
    train_loader = get_data_loader(train_dataset, config["batch_size"])
    val_loader = get_data_loader(val_dataset, config["batch_size"])

    if config["use_manual_hyperparameters"]:
        train_model(
            config,
            device,
            train_loader,
            val_loader,
            train_cell_id_mapping,
            val_cell_id_mapping,
            num_labels_list,
        )
    else:
        objective_with_config_and_data = functools.partial(
            objective,
            train_loader=train_loader,
            val_loader=val_loader,
            train_cell_id_mapping=train_cell_id_mapping,
            val_cell_id_mapping=val_cell_id_mapping,
            num_labels_list=num_labels_list,
            config=config,
            device=device,
        )

        study = optuna.create_study(
            direction="minimize",  # Minimize validation loss
            study_name=config["study_name"],
            # storage=config["storage"],
            load_if_exists=True,
        )

        study.optimize(objective_with_config_and_data, n_trials=config["n_trials"])

        # After finding the best trial
        best_params = study.best_trial.params
        best_task_weights = study.best_trial.user_attrs["task_weights"]
        print("Saving the best model and its hyperparameters...")

        # Saving model as before
        best_model = GeneformerMultiTask(
            config["pretrained_path"],
            num_labels_list,
            dropout_rate=best_params["dropout_rate"],
            use_task_weights=config["use_task_weights"],
            task_weights=best_task_weights,
        )

        # Get the best model state dictionary
        best_model_state_dict = study.best_trial.user_attrs["model_state_dict"]

        # Remove the "module." prefix from the state dictionary keys if present
        best_model_state_dict = {
            k.replace("module.", ""): v for k, v in best_model_state_dict.items()
        }

        # Load the modified state dictionary into the model, skipping unexpected keys
        best_model.load_state_dict(best_model_state_dict, strict=False)

        model_save_directory = os.path.join(
            config["model_save_path"], "GeneformerMultiTask"
        )
        save_model(best_model, model_save_directory)

        # Additionally, save the best hyperparameters and task weights
        hyperparams_path = os.path.join(model_save_directory, "hyperparameters.json")

        with open(hyperparams_path, "w") as f:
            json.dump({**best_params, "task_weights": best_task_weights}, f)
        print(f"Best hyperparameters and task weights saved to {hyperparams_path}")
