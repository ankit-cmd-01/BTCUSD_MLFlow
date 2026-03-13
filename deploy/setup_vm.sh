#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

sudo apt update
sudo apt install -y python3 python3-venv python3-pip nginx

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [[ ! -f ".env" ]]; then
  cp .env.example .env
fi

chmod +x deploy/run_vm.sh

echo "VM dependencies installed."
echo "Next steps:"
echo "1. Edit .env if needed"
echo "2. Run: ./deploy/run_vm.sh"
echo "3. For background service, use deploy/systemd/btcusd-mlflow.service"
