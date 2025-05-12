from hydra.core.config_store import ConfigStore

from config.hydra_configs.optimizers.adamw import AdamWOptimizerConfig
from config.hydra_configs.optimizers.dummy_optimizer import DummyOptimizerConfig

cs = ConfigStore.instance()
cs.store(group="optimizer", name="dummy", node=DummyOptimizerConfig)
cs.store(group="optimizer", name="adamw", node=AdamWOptimizerConfig)
