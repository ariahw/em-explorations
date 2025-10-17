# pick the right tarball for your CPU
ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
  FILE="codex-x86_64-unknown-linux-musl.tar.gz"
else
  FILE="codex-aarch64-unknown-linux-musl.tar.gz"
fi

curl -fLO "https://github.com/openai/codex/releases/latest/download/$FILE"

TMP=$(mktemp -d)
tar -C "$TMP" --no-same-owner -xzf "$FILE"

# put the binary on your PATH (creates dir if needed)
install -Dm755 "$TMP"/codex-* "$HOME/.local/bin/codex"

# ensure PATH covers ~/.local/bin for future shells
grep -q 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc || \
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"

# cleanup
rm -rf "$TMP" "$FILE"

# sanity checks
which codex
codex --version