import gc
import torch

# Clear the cache
gc.collect()
torch.cuda.empty_cache()

# Clear the cache
try:
    torch.cuda.ipc_collect()
except:
    pass

try:
    # Then let PyTorch tear down the process group, if vLLM initialized it
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()  # or dist.shutdown() on recent PyTorch
except AssertionError:
    pass

try:
    import ctypes
    ctypes.CDLL("libc.so.6").malloc_trim(0)
except OSError: 
    pass