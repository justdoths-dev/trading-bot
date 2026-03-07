# Trading Bot

Automated crypto trading bot with multi-timeframe market analysis.

---

# Project Goal

Build a trading system that can evolve from scalping to intraday and swing trading.

Timeframe expansion order:

1m → 5m → 15m → 1h → 4h → 1d

The system will eventually include:

- Market data collection
- Indicator calculation
- Strategy engine
- Risk management
- Trade execution
- Telegram reporting

---

# Current Progress

## Completed

- [x] WSL development environment
- [x] Python virtual environment
- [x] GitHub repository setup
- [x] BinanceMarketDataClient implemented
- [x] MultiTimeframeLoader implemented
- [x] Multi timeframe candle loading
- [x] 1m / 5m / 15m / 1h / 4h / 1d data collection
- [x] Base project architecture

## In Progress

- [ ] Indicator Layer

## Next Tasks

- [ ] RSI indicator
- [ ] EMA indicator
- [ ] MACD indicator
- [ ] Bollinger Bands
- [ ] ATR indicator
- [ ] Indicator Engine
- [ ] Strategy Engine integration

---

## Project Structure

```
trading-bot
│
├── src
│   ├── config
│   ├── data
│   ├── exchange
│   ├── services
│   ├── strategy
│   └── utils
│
├── main.py
├── requirements.txt
├── .env (not tracked)
└── .gitignore
```

---

## How To Run

### Activate Environment

```bash
cd ~/workspace/bots/trading-bot
source venv/bin/activate
```

### Run Bot

```bash
python main.py
```

## Daily Development Start

When starting development:

1. Open VS Code
2. Connect WSL
3. Activate virtual environment
4. Run `python main.py`
5. Continue development

---

## Git Workflow

After code changes:

```bash
git add .
git commit -m "update"
git push
```

## Security

Never upload these files:
.env
venv/
These are ignored by `.gitignore`.