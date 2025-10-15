# RUNPOD SETUP FILE
# ASSUMES YOU HAVE A GIT KEY SAVED IN workspace/.ssh/id_ed25519


export IS_RUNPOD_ENV=true

# Load environment variables from .env file if it exists
if [ -f .env.runpod ]; then
    echo "Loading environment variables from .env.runpod file"
    set -a  # Automatically export all variables
    source .env.runpod
    set +a  # Turn off automatic export
    echo "Loaded environment variables from .env file"
else
    echo ".env.runpod file missing! Exiting..."
    exit 2
fi

# Run installation of basic packages
apt-get update
apt-get install -y vim
apt-get install -y git
apt-get install -y tmux

# Ensure ssh dir and permissions
mkdir -p ~/.ssh
cp $NFS_DIR/.ssh/id_ed25519 ~/.ssh/id_ed25519
chmod 600 ~/.ssh/id_ed25519

# Optional: avoid strict host key checking
echo -e "Host *\n\tStrictHostKeyChecking no\n" > ~/.ssh/config

# Add GitHub as known host
ssh-keyscan github.com >> ~/.ssh/known_hosts

# Check if directory already exists before cloning
if [ ! -d "$NFS_DIR/$GIT_REPO_NAME" ]; then
    git clone git@github.com:ariahw/$GIT_REPO_NAME.git $NFS_DIR/$GIT_REPO_NAME
else
    echo "Directory $NFS_DIR/$GIT_REPO_NAME already exists, skipping git clone"
fi

# Load other environment variables
cd $NFS_DIR/$GIT_REPO_NAME
source setup.sh

# Unsloth will use a local file cache which is slower for runpod, change to tmp dir
if [ ! -e "$NFS_DIR/$GIT_REPO_NAME/unsloth_compiled_cache" ]; then
    ln -s $LOCAL_SSD_DIR/unsloth_compiled_cache $NFS_DIR/$GIT_REPO_NAME/unsloth_compiled_cache
fi

# Install uv
pip install uv
uv venv $VENV_DIR
source $VENV_DIR/bin/activate
uv sync --active

# Create jupyter kernel
uv run --active python -m ipykernel install --user --name sl-venv --display-name "Python (sl-venv)"

export TMUX_SESSION_NAME=workworkwork

# Create a new tmux session
tmux new -s $TMUX_SESSION_NAME
tmux attach -d -t "$TMUX_SESSION_NAME"

# NOTE: User needs to manually activate the venv in order to run uv commands using --active flag

