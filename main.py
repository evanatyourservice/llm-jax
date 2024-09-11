import tyro

from configs import TrainConfig, get_default_config
from train import main


if __name__ == "__main__":
    config = tyro.cli(TrainConfig, default=get_default_config(), use_underscores=True)
    main(config)
