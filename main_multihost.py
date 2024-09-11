import tyro
import jax

from configs import TrainConfig, get_default_config
from train import main


if __name__ == "__main__":
    jax.distributed.initialize()

    config = tyro.cli(TrainConfig, default=get_default_config(), use_underscores=True)
    main(config)
