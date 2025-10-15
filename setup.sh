# Setup file

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    set -a  # Automatically export all variables
    source .env
    set +a  # Turn off automatic export
    echo "Loaded environment variables from .env file"
else
    echo ".env file not found, skipping environment variable loading"
fi

# Recommended settings
export WANDB_LOG_MODEL=false # Prevent sending model to weights and biases, prefer local storage
export WANDB_START_METHOD=thread # Use thread instead of process to avoid issues with wandb
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Recommended settings for flash attention install
# Change these settings if not for Ampere RTX A6000 is CC 8.6
# 5090 = Blackwell, compute capability **12.0** (sm_120)

# 1) Target the right GPU arch when compiling CUDA extensions:
export TORCH_CUDA_ARCH_LIST="12.0"     # or "12.0+PTX" if you want a PTX fallback

# 2) Match PyTorch’s C++ ABI when building native extensions (flash-attn, etc.)
#    (_GLIBCXX_USE_CXX11_ABI is a **compile-time macro**, so pass it via CXXFLAGS)
export CXX11_ABI=$(python - <<'PY'
import torch, sys
sys.stdout.write("1" if torch._C._GLIBCXX_USE_CXX11_ABI else "0")
PY
)
export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=$CXX11_ABI ${CXXFLAGS:-}"
export USE_CXX11_ABI="$CXX11_ABI"      # some build scripts read this
export _GLIBCXX_USE_CXX11_ABI="$CXX11_ABI"  # harmless; only effective if the build script consumes it


export MAX_JOBS=8


# Install uv
pip install uv

# Sync uv
uv sync