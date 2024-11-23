import sys
from typing import Optional

from torch.utils._config_module import Config, install_config_module


e_bool = True
e_int = 1
e_float = 1.0
e_string = "string"
e_list = [1]
e_set = {1}
e_tuple = (1,)
e_dict = {1: 2}
e_none: Optional[bool] = None
e_optional: Optional[bool] = True
e_ignored = True
_e_ignored = True
magic_cache_config_ignored = True
# [@compile_ignored: debug]
e_compile_ignored = True
e_config = Config(default=True)
e_jk = Config(justknob="does_not_exist")
e_jk_false = Config(justknob="does_not_exist", default=False)
e_env_default = Config(env_name_default="ENV_TRUE", default=False)
e_env_default_FALSE = Config(env_name_default="ENV_FALSE", default=True)
e_env_force = Config(env_name_force="ENV_TRUE", default=False)


class nested:
    e_bool = True


_cache_config_ignore_prefix = ["magic_cache_config"]
_save_config_ignore = ["e_ignored"]

install_config_module(sys.modules[__name__])