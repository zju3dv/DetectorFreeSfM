from pytorch_lightning import LightningModule, Callback, Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import LightningLoggerBase
import torch

import hydra
from omegaconf import DictConfig
from typing import List
from src.utils import template_utils as utils

import warnings

warnings.filterwarnings("ignore")


def train(config: DictConfig):
    if config["print_config"]:
        utils.print_config(config)

    if "seed" in config:
        seed_everything(config["seed"])

    # scale lr and warmup-step automatically
    if not isinstance(config.trainer.gpus, (int, str)):
        # List type
        _n_gpus = len(config.trainer.gpus)
    else:
        _n_gpus = (
            int(config.trainer.gpus)
            if "," not in str(config.trainer.gpus)
            else len([num for num in config.trainer.gpus.split(",") if num != ""])
        )
        _n_gpus = _n_gpus if _n_gpus != -1 else torch.cuda.device_count()
    config.model.trainer.world_size = _n_gpus * config.trainer.num_nodes
    true_batch_size = config.model.trainer.world_size * config.datamodule.batch_size
    _scaling = true_batch_size / config.model.trainer.canonical_bs
    # config.model.trainer.scaling= _scaling
    config.model.trainer.true_lr = config.model.trainer.canonical_lr * _scaling

    # Init PyTorch Lightning model ⚡
    model: LightningModule = hydra.utils.instantiate(config["model"])

    # Init PyTorch Lightning datamodule ⚡
    datamodule: LightningModule = hydra.utils.instantiate(config["datamodule"])
    # datamodule.setup()

    # Init PyTorch Lightning callbacks ⚡
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init PyTorch Lightning loggers ⚡
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init PyTorch Lightning trainer ⚡
    trainer: Trainer = hydra.utils.instantiate(
        # config["trainer"], callbacks=callbacks, logger=logger, plugins=DDPPlugin(find_unused_parameters=False)
        config["trainer"], callbacks=callbacks, logger=logger
    )

    # Send some parameters from config to all lightning loggers
    utils.log_hparams_to_all_loggers(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    trainer.fit(model=model, datamodule=datamodule)

    # Make sure everything closed properly
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Return best achieved metric score for optuna
    optimized_metric = config.get("optimized_metric", None)
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]


@hydra.main(config_path="hydra_training_configs/", config_name="config.yaml")
def main(config: DictConfig):
    return train(config)


if __name__ == "__main__":
   main()
