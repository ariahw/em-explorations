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
# Blackwell RTX 6000 PRO has compute capability 12.0 (sm_120)
# Override with GPU_COMPUTE_CAP or TORCH_CUDA_ARCH_LIST if you need something else

detect_compute_capability() {
    # Prefer explicit override
    if [ -n "${GPU_COMPUTE_CAP:-}" ]; then
        echo "${GPU_COMPUTE_CAP}"
        return 0
    fi

    # Try to auto-detect via nvidia-smi (one entry per GPU)
    if command -v nvidia-smi >/dev/null 2>&1; then
        local caps
        caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' ' | tr '\n' ';' | sed 's/;$//')
        if [ -n "$caps" ]; then
            echo "$caps"
            return 0
        fi
    fi

    return 1
}

if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    if gpu_cap_list=$(detect_compute_capability); then
        export TORCH_CUDA_ARCH_LIST="$gpu_cap_list"
        echo "Detected compute capability: $TORCH_CUDA_ARCH_LIST"
    else
        export TORCH_CUDA_ARCH_LIST="12.0"  # default for Blackwell RTX 6000 PRO
        echo "Falling back to TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST (Blackwell default)"
    fi
else
    echo "Using existing TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
fi
# If you need PTX fallback, append '+PTX' (e.g. export TORCH_CUDA_ARCH_LIST=\"12.0+PTX\")

# 2) Match PyTorch’s C++ ABI when building native extensions (flash-attn, etc.)
#    (_GLIBCXX_USE_CXX11_ABI is a **compile-time macro**, so pass it via CXXFLAGS)
CXX11_ABI=$(python - <<'PY' 2>/dev/null || true
import sys
try:
    import torch
except Exception:
    sys.stdout.write("")
else:
    sys.stdout.write("1" if torch._C._GLIBCXX_USE_CXX11_ABI else "0")
PY
)

if [ -z "$CXX11_ABI" ]; then
    # Default to the modern ABI when torch is not importable yet.
    CXX11_ABI=1
    echo "Torch not installed yet; defaulting _GLIBCXX_USE_CXX11_ABI=$CXX11_ABI"
else
    echo "Configured _GLIBCXX_USE_CXX11_ABI=$CXX11_ABI based on installed torch"
fi

export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=$CXX11_ABI ${CXXFLAGS:-}"
export USE_CXX11_ABI="$CXX11_ABI"      # some build scripts read this
export _GLIBCXX_USE_CXX11_ABI="$CXX11_ABI"  # harmless; only effective if the build script consumes it


export MAX_JOBS=8


# Install uv
pip install uv

# Sync uv
uv sync
