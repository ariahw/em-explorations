from pathlib import Path
from unsloth_zoo.compiler_replacements import compiler_replacements


# Export vars
# UNSLOTH_COMPILE_LOCATION =
# UNSLOTH_COMPILE_OVERWRITE=0

cache_dir = Path("/workspace/em-explorations/unsloth_compiled_cache")
source = (cache_dir / "UnslothGRPOTrainer.py").read_text()  # or build it however you like
source = source.replace("buggy_line", "your_fixed_line")     # apply your modifications
compiler_replacements["UnslothGRPOTrainer"] = source