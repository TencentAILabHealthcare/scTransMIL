import optuna
from optuna.integration import TensorBoardCallback


def save_trial_callback(study, trial, trials_result_path):
    with open(trials_result_path, "a") as f:
        f.write(
            f"Trial {trial.number}: Value (F1 Macro): {trial.value}, Params: {trial.params}\n"
        )


def create_optuna_study(objective, n_trials, trials_result_path, tensorboard_log_dir):
    study = optuna.create_study(direction="maximize")

    # init TensorBoard callback
    tensorboard_callback = TensorBoardCallback(
        dirname=tensorboard_log_dir, metric_name="F1 Macro"
    )

    # callback and TensorBoard callback
    callbacks = [
        lambda study, trial: save_trial_callback(study, trial, trials_result_path),
        tensorboard_callback,
    ]

    study.optimize(objective, n_trials=n_trials, callbacks=callbacks)
    return study
