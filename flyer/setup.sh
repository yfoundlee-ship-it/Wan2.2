#!/usr/bin/env bash
# One-time setup for the animated flyer renderer.
#   ./setup.sh          install requirements + a local ffmpeg if needed
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}

echo "==> python venv"
"$PY" -m venv .venv 2>/dev/null || true
./.venv/bin/pip install --quiet --upgrade pip
./.venv/bin/pip install --quiet pillow numpy

if command -v ffmpeg >/dev/null 2>&1; then
  echo "==> ffmpeg found: $(command -v ffmpeg)"
  exit 0
fi

echo "==> no system ffmpeg - installing a static build via npm"
mkdir -p .tools
( cd .tools && npm init -y >/dev/null 2>&1 && npm i --no-audit --no-fund @ffmpeg-installer/ffmpeg >/dev/null 2>&1 )

FF=".tools/node_modules/@ffmpeg-installer/linux-x64/ffmpeg"
if [ -x "$FF" ]; then
  ln -sf "$(pwd)/$FF" .tools/ffmpeg
  echo "==> ffmpeg installed at .tools/ffmpeg (render_flyer.py finds it automatically)"
else
  echo "!! could not fetch a static ffmpeg. Install one and re-run, e.g.:"
  echo "   sudo apt install ffmpeg     (or set \$FFMPEG to a binary)"
fi
