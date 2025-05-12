from hydra.core.config_store import ConfigStore

from config.hydra_configs.accelerators.debug_accelerator import DebugAcceleratorConfig
from config.hydra_configs.accelerators.deepspeed_accelerator import DeepSpeedAcceleratorConfig

ACCELERATOR_GROUP_NAME = "accelerator"

cs = ConfigStore.instance()
cs.store(group=ACCELERATOR_GROUP_NAME, name="deepspeed", node=DeepSpeedAcceleratorConfig)
cs.store(group=ACCELERATOR_GROUP_NAME, name="debug", node=DebugAcceleratorConfig)
