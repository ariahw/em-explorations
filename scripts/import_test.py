# import unsloth
import fire
import torch

#!/usr/bin/env python3
# fa_wheel_finder.py
import sys, platform, re, argparse, urllib.request, urllib.error

def safe_parse_torch_mm_patch(torch_version: str):
    """
    Returns (major.minor, major.minor.patch) as strings from torch.__version__ like '2.8.0+cu128'.
    """
    core = re.split(r'[+-]', torch_version)[0]  # e.g. '2.8.0'
    m = re.match(r'^(\d+)\.(\d+)(?:\.(\d+))?$', core)
    if not m:
        # fallback: best effort
        parts = core.split('.')
        while len(parts) < 3: parts.append('0')
        return f'{parts[0]}.{parts[1]}', f'{parts[0]}.{parts[1]}.{parts[2]}'
    MAJ, MIN, PAT = m.group(1), m.group(2), (m.group(3) or '0')
    return f'{MAJ}.{MIN}', f'{MAJ}.{MIN}.{PAT}'

def detect_env():
    try:
        import torch
    except Exception as e:
        print("ERROR: PyTorch is not importable in this environment.\n", e, file=sys.stderr)
        sys.exit(1)

    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"

    # Platform tag (FlashAttention wheels typically use plain 'linux_x86_64', 'linux_aarch64', or 'win_amd64')
    machine = platform.machine().lower()
    system = platform.system()
    if system == "Linux":
        plat_tag = f"linux_{'x86_64' if machine in ('x86_64','amd64') else machine}"
    elif system == "Windows":
        plat_tag = "win_amd64" if machine in ("x86_64", "amd64") else f"win_{machine}"
    elif system == "Darwin":
        # Not generally provided upstream; will warn below.
        plat_tag = f"macosx_{machine}"
    else:
        plat_tag = f"{sys.platform}_{machine}"

    torch_ver_raw = getattr(__import__('torch'), "__version__", "")
    torch_mm, torch_mmp = safe_parse_torch_mm_patch(torch_ver_raw)

    cuda = getattr(torch.version, "cuda", None)  # e.g. '12.8' or None
    hip = getattr(torch.version, "hip", None)    # ROCm version or None

    cu_major = cu_full = None
    if cuda:
        # Many Linux wheels encode CUDA as 'cu12', but some wheels (esp. Windows/community) use 'cu124' etc.
        parts = cuda.split(".")
        if len(parts) >= 2:
            cu_major = f"cu{parts[0]}"
            cu_full  = f"cu{parts[0]}{parts[1]}"
        else:
            cu_major = f"cu{parts[0]}"
            cu_full  = cu_major

    # Detect PyTorch's C++11 ABI setting (Linux)
    abi_str = "FALSE"
    abi_bool = None
    try:
        from torch import compiled_with_cxx11_abi
        abi_bool = compiled_with_cxx11_abi()
    except Exception:
        try:
            abi_bool = bool(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI"))
        except Exception:
            abi_bool = None
    if abi_bool is not None:
        abi_str = "TRUE" if abi_bool else "FALSE"

    return {
        "system": system,
        "py_tag": py_tag,
        "plat_tag": plat_tag,
        "torch_ver_raw": torch_ver_raw,
        "torch_mm": torch_mm,
        "torch_mmp": torch_mmp,
        "cuda": cuda,
        "hip": hip,
        "cu_major": cu_major,
        "cu_full": cu_full,
        "abi_str": abi_str,
    }

def build_candidates(fa_version: str, env: dict):
    """
    Return wheel filename candidates in most-likely order.
    We generate both naming styles seen in the wild:
      - CUDA major only (e.g., cu12)
      - CUDA major+minor squashed (e.g., cu128)
    And torch version with and without patch (torch2.8 vs torch2.8.0).
    """
    base = f"flash_attn-{fa_version}"
    py = env["py_tag"]
    plat = env["plat_tag"]
    abi = env["abi_str"]

    cands = []
    if env["cuda"]:
        cu_tags = []
        if env["cu_major"]:
            cu_tags.append(env["cu_major"])
        if env["cu_full"] and env["cu_full"] != env["cu_major"]:
            cu_tags.append(env["cu_full"])
        torch_tags = [env["torch_mm"], env["torch_mmp"]]

        for cu in cu_tags:
            for ttag in torch_tags:
                cands.append(f"{base}+{cu}torch{ttag}cxx11abi{abi}-{py}-{py}-{plat}.whl")
    else:
        # CPU-only / ROCm builds generally don't have precompiled CUDA wheels
        cands.append(f"{base}-{py}-{py}-{plat}.whl")

    # Deduplicate in order
    seen, out = set(), []
    for s in cands:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out

def url_exists(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=6) as resp:
            return 200 <= resp.status < 400
    except Exception:
        # Fallback: tiny GET with range to dodge full download
        try:
            req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                return 200 <= resp.status < 400
        except Exception:
            return False

def main():
    ap = argparse.ArgumentParser(description="Find the matching FlashAttention wheel for your environment.")
    ap.add_argument("--fa", "--flash-attn-version", dest="fa", default="2.7.4.post1",
                    help="FlashAttention version tag (e.g. 2.8.3, 2.7.4.post1). Default: %(default)s")
    ap.add_argument("--print-all", action="store_true", help="Print all candidate filenames (without checking GitHub).")
    args = ap.parse_args()

    env = detect_env()

    print("# Detected environment")
    print(f"Python tag:          {env['py_tag']}")
    print(f"Platform tag:        {env['plat_tag']}")
    print(f"torch.__version__:   {env['torch_ver_raw']}")
    print(f"torch.version.cuda:  {env['cuda']}")
    if env["hip"]:
        print(f"torch.version.hip:   {env['hip']} (ROCm)")
    print(f"C++11 ABI (PyTorch): {env['abi_str']}")
    if env["system"] == "Darwin":
        print("\nNOTE: macOS isn’t officially supported for CUDA wheels; you’ll likely need to build from source.")
    if not env["cuda"] and not env["hip"]:
        print("\nWARNING: Your torch build appears CPU-only; CUDA wheels won’t work (try building from source).")

    candidates = build_candidates(args.fa, env)
    base = f"https://github.com/Dao-AILab/flash-attention/releases/download/v{args.fa}/"

    print("\n# Candidate wheel filenames (most likely first):")
    for fn in candidates:
        print("  ", fn)

    if args.print_all:
        print("\n# Candidate URLs:")
        for fn in candidates:
            print("  ", base + fn)
        return

    # Probe GitHub to pick the first URL that exists
    print("\n# Probing GitHub release for a matching wheel…")
    for fn in candidates:
        url = base + fn
        if url_exists(url):
            print("MATCH FOUND:")
            print("  Filename:", fn)
            print("  URL:     ", url)
            print("\nInstall with:\n  pip install \"" + url + "\"")
            break
    else:
        print("No candidate found at that release tag.")
        print("Try another FlashAttention version with --fa (e.g., 2.8.3) or build from source:")
        print("  pip install flash-attn --no-build-isolation")


def test_flash_attn_import(engine = 'vllm'):
    if engine == 'transformers':
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct", 
            torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="flash_attention_2"
        )
    elif engine == 'vllm':
        from vllm import LLM
        llm = LLM(
            model="meta-llama/Llama-3.1-8B-Instruct", 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
    elif engine == 'unsloth':
        import unsloth
        model = unsloth.FastLanguageModel.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            torch_dtype=torch.bfloat16, device_map="auto",
            fast_inference=True,
        )

    print(model)


def check_imports():
    try:
        import torch
        print("torch", torch.__version__, "cuda", torch.version.cuda, "cap", torch.cuda.get_device_capability())
    except Exception as e:
        print("torch import failed:", e)

    try:
        import flash_attn
        print("flash-attn version", flash_attn.__version__)
    except Exception as e:
        print("flash-attn import failed:", e)

    try:
        import vllm
        print("vllm version", vllm.__version__)
    except Exception as e:
        print("vllm import failed:", e)

    try:
        import triton
        print("triton import OK")
        print("triton version", triton.__version__)
    except Exception as e:
        print("triton import failed:", e)

    try:
        import transformers
        print("transformers version", transformers.__version__)
    except Exception as e:
        print("transformers import failed:", e)


if __name__ == "__main__":
    # main()
    fire.Fire(check_imports)