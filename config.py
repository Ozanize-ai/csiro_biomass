"""
Config loader.

Reads the ``CSIRO_CONFIG`` environment variable to determine which experiment
config to load from ``configs/``. Defaults to ``lvd_1024`` if not set.

Usage from an entry-point script (e.g. train.py):

    import os, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="lvd_1024")
    args = parser.parse_args()
    os.environ["CSIRO_CONFIG"] = args.config
    import config  # must come after setting the env var

Other modules keep using ``import config`` and get whatever was loaded when
the entry point first imported this module.

To run a helper script directly against a non-default config, set the env
var before invoking Python:

    set CSIRO_CONFIG=lvd_1024 && python your_script.py   # Windows
    CSIRO_CONFIG=lvd_1024 python your_script.py          # Linux/Mac
"""

import os
import importlib

_cfg_name = os.environ.get("CSIRO_CONFIG", "lvd_1024")
_module_path = f"configs.config_{_cfg_name}"

try:
    _mod = importlib.import_module(_module_path)
except ImportError as e:
    raise ImportError(
        f"Could not load config '{_cfg_name}' from {_module_path}. "
        f"Make sure configs/config_{_cfg_name}.py exists. "
        f"Available configs: see configs/ folder. Original error: {e}"
    )

# Re-export every public name from the loaded config module so both
# ``from config import X`` and ``config.X`` work throughout the codebase.
for _k in dir(_mod):
    if not _k.startswith("_"):
        globals()[_k] = getattr(_mod, _k)

LOADED_CONFIG_NAME = _cfg_name

# Print the loaded config name once. Multiprocessing DataLoader workers also
# import this module, so we guard with an env var sentinel to keep the log
# clean rather than seeing the print duplicated per worker.
if os.environ.get("CSIRO_CONFIG_PRINTED") != "1":
    print(f"[config] Loaded: configs/config_{_cfg_name}.py  (RUN_NAME={_mod.RUN_NAME})")
    os.environ["CSIRO_CONFIG_PRINTED"] = "1"
