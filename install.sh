#!/usr/bin/env bash
set -euo pipefail

python3 -m venv venv
. venv/bin/activate

python -m pip install --upgrade pip
python -m pip install pillow pyvips dacite psutil

      DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
   BASHRC="$HOME/.bashrc"
PATH_LINE="export PATH=\"\$PATH:$DIR\""

chmod +x -- "$DIR/anime-ultrascale"

grep -Fqx -- "$PATH_LINE" "$BASHRC" ||
    printf '\n%s\n' "$PATH_LINE" >> "$BASHRC"

source "$BASHRC"
