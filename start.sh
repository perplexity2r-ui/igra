#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PAPER_JAR="${PAPER_JAR:-paper.jar}"
PAPER_URL="https://api.papermc.io/v2/projects/paper/versions/1.16.5/builds/794/downloads/paper-1.16.5-794.jar"

if [[ ! -f "$PAPER_JAR" ]]; then
  echo "[start.sh] '$PAPER_JAR' not found. Downloading Paper 1.16.5 build 794..."
  curl -fsSL "$PAPER_URL" -o "$PAPER_JAR"
fi

echo "[start.sh] Starting Paper 1.16.5 server on port 25565..."
java -Xms1G -Xmx2G -jar "$PAPER_JAR" --nogui &
SERVER_PID=$!

echo "[start.sh] Waiting 10s for Paper server initialization..."
sleep 10

echo "[start.sh] Installing/checking Node dependencies and starting Mineflayer bots..."
(cd bot && npm install --no-audit --no-fund && node bot.js) &
BOT_PID=$!

cleanup() {
  echo "[start.sh] Shutting down bots and server..."
  kill "$BOT_PID" 2>/dev/null || true
  kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait "$SERVER_PID" "$BOT_PID"
