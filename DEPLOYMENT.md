# Trading Bot Deployment and Operations (VPS)

This document describes how to deploy and operate the trading bot on a Linux VPS.

## 1) Clone on VPS

```bash
cd /home/dev/workspace/bots
git clone <YOUR_REPO_URL> trading-bot
cd /home/dev/workspace/bots/trading-bot
```

## 2) Python venv setup

Use Python 3.12 and create a local virtual environment.

```bash
cd /home/dev/workspace/bots/trading-bot
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Quick check:

```bash
.venv/bin/python --version
.venv/bin/python main.py
```

## 3) Create .env

Create runtime secrets/config file:

```bash
cd /home/dev/workspace/bots/trading-bot
cp .env.example .env
nano .env
```

Required for the current paper/public-data deployment:
- OPENAI_API_KEY
- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHAT_ID

Optional for the current public-data / paper-mode setup:
- BINANCE_API_KEY
- BINANCE_API_SECRET

Lock permissions:

```bash
chmod 600 .env
```

## 4) Install systemd service

Copy and enable the example service.

```bash
sudo cp ops/trading-bot.service.example /etc/systemd/system/trading-bot.service
sudo systemctl daemon-reload
sudo systemctl enable trading-bot
sudo systemctl start trading-bot
```

Check status and logs:

```bash
sudo systemctl status trading-bot
journalctl -u trading-bot -n 100 --no-pager
```

## 5) Configure logrotate

Install and register logrotate config for `logs/trade_analysis.jsonl`.

```bash
sudo cp ops/logrotate.trading-bot.example /etc/logrotate.d/trading-bot
sudo logrotate -d /etc/logrotate.d/trading-bot
sudo logrotate -f /etc/logrotate.d/trading-bot
```

## 6) Reboot verification

Validate service auto-start after reboot.

```bash
sudo reboot
```

After reconnect:

```bash
sudo systemctl status trading-bot
journalctl -u trading-bot -n 100 --no-pager
```

## 7) Operational checks

Run these checks regularly:

```bash
# Service health
sudo systemctl is-active trading-bot

# Recent logs
journalctl -u trading-bot -n 200 --no-pager

# App output log file
ls -lh /home/dev/workspace/bots/trading-bot/logs/trade_analysis.jsonl
tail -n 20 /home/dev/workspace/bots/trading-bot/logs/trade_analysis.jsonl

# Verify scheduler loop still running
ps -ef | grep "main.py" | grep -v grep
```

Optional restart when changing code/config:

```bash
cd /home/dev/workspace/bots/trading-bot
git pull
source .venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart trading-bot
sudo systemctl status trading-bot
```
