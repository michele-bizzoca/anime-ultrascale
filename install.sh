#!/usr/bin/env bash
set -uo pipefail

DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASHRC="$HOME/.bashrc"
PATH_LINE="export PATH=\"\$PATH:$DIR/bin\""

cd "$DIR"

python3 -m venv venv
source venv/bin/activate
venv/bin/python3 -m pip install --ignore-installed --no-user .
rm -rf ./*.egg-info

chmod +x -- "bin/anime-ultrascale"

grep -Fqx -- "$PATH_LINE" "$BASHRC" ||
    printf '\n%s\n' "$PATH_LINE" >> "$BASHRC"

source "$BASHRC"
