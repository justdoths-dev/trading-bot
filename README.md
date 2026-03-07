# Trading Bot

Automated crypto trading system with multi-timeframe analysis, AI interpretation, logging, and Telegram reporting.

The project is designed to evolve from a research prototype into a modular trading system capable of supporting scalping, intraday, and swing trading strategies.

---

# Project Goal

Build a scalable crypto trading system that can progressively expand across multiple time horizons.

Timeframe expansion order:

1m → 5m → 15m → 1h → 4h → 1d

The system will eventually include:

- Market data collection
- Indicator calculation
- Strategy engine
- Risk management
- Trade execution
- AI interpretation
- Trade analysis logging
- Telegram reporting
- Strategy separation (scalp / intraday / swing)

---

# Current System Architecture

Current trading pipeline:

Market Data  
→ Indicator Engine  
→ Strategy Engine  
→ Risk Manager  
→ Execution Engine  
→ AI Interpretation Layer  
→ Trade Analysis Logger  
→ Telegram Formatter  
→ Telegram Sender  

The rule-based engine remains the primary decision system.

The AI layer acts as a **market interpreter**, not a trade decision maker.

---

# Implemented Features

## Development Environment

- WSL development environment
- Python virtual environment
- GitHub repository workflow
- `.env` configuration management

---

## Market Data Layer

Components:

- `BinanceMarketDataClient`
- `MultiTimeframeLoader`

Capabilities:

- Binance API integration
- Multi-timeframe OHLCV loading

Supported timeframes:

```

1m
5m
15m
1h
4h
1d

```

---

## Indicator Layer

Implemented indicators:

- RSI
- EMA (20 / 50)
- MACD
- ATR

Indicator Engine responsibilities:

- Enrich raw market data
- Preserve OHLCV structure
- Append calculated indicator columns

---

## Strategy Engine

Strategy architecture uses layered validation.

Layers:

Bias Layer  
→ evaluates higher timeframe alignment

Setup Layer  
→ evaluates mid-timeframe structure

Trigger Layer  
→ confirms lower timeframe entry timing

Possible outputs:

```

long
short
watchlist_long
watchlist_short
no_signal

```

---

## Risk Management

Risk Manager calculates:

- ATR-based stop loss
- ATR-based take profit
- Risk / reward ratio
- Volatility filtering

Output:

```

execution_allowed
entry_price
stop_loss
take_profit
risk_reward_ratio

```

---

## Execution Engine

Current mode:

```

paper

```

Responsibilities:

- Convert strategy signals into trade plans
- Maintain deterministic rule-based execution decisions

---

## AI Interpretation Layer

Components:

- `AIPayloadBuilder`
- `AIPromptBuilder`
- `OpenAIAnalyzer`
- `AIService`

AI responsibilities:

- Interpret rule-based output
- Explain market structure
- Identify signal bottlenecks
- Generate trade briefings

The AI layer **never overrides rule-based execution decisions**.

---

## Logging System

Trade analysis results are stored using JSONL logging.

Log file:

```

logs/trade_analysis.jsonl

```

Each record contains:

- rule engine output
- risk evaluation
- execution decision
- AI interpretation
- rule vs AI alignment

This enables:

- signal analytics
- AI vs rule comparison
- debugging of decision flow

---

## Telegram Reporting

Modules:

- `TelegramFormatter`
- `TelegramSender`

Workflow:

AI Analysis  
→ TelegramFormatter  
→ TelegramSender  
→ Telegram Chat

Example message:

```

[BTCUSDT]

Signal: no_signal
Bias: neutral_conflict
Action: hold

Key Bottlenecks

* Higher timeframe bias conflict

Briefing

* No clear entry conditions
* Wait for confirmation

```

---

# Project Structure

```

trading-bot
│
├── src
│   ├── ai
│   ├── config
│   ├── data
│   ├── exchange
│   ├── execution
│   ├── indicators
│   ├── risk
│   ├── storage
│   ├── strategy
│   └── telegram
│
├── logs
│
├── main.py
├── requirements.txt
├── .env (not tracked)
└── .gitignore

````

---

# How To Run

Activate environment:

```bash
cd ~/workspace/bots/trading-bot
source venv/bin/activate
````

Run bot:

```bash
python main.py
```

---

# Daily Development Start

Typical workflow:

1. Open VS Code
2. Connect to WSL
3. Activate virtual environment
4. Run the bot

```bash
python main.py
```

5. Continue development

---

# Git Workflow

After making changes:

```bash
git add .
git commit -m "message"
git push
```

Recommended commit prefixes:

```
feat: new feature
fix: bug fix
refactor: code improvement
docs: documentation update
chore: maintenance change
```

---

# Security

Never upload these files:

```
.env
venv/
logs/
```

Sensitive credentials must always remain outside the repository.

Example `.env`:

```
BINANCE_API_KEY=
BINANCE_API_SECRET=
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
```

---

# Future Development Roadmap

Planned improvements:

### Strategy Layer

Strategy separation:

```
scalping
intraday
swing
```

Each strategy will operate with different timeframe emphasis.

---

### Indicator Expansion

Planned indicators:

* Bollinger Bands
* Volume analysis
* Trend strength filters

---

### AI Layer Improvements

Future upgrades:

* AI confidence scoring
* rule vs AI disagreement detection
* trade explanation analytics

---

### Execution Layer

Planned features:

* live trade execution
* exchange order management
* execution safety checks

---

### Monitoring & Reporting

Future improvements:

* Telegram automation improvements
* trade performance dashboards
* strategy performance tracking

---

# Project Status

Current stage:

Research-grade automated trading analysis system with AI interpretation and reporting.

Estimated completion:

~80%

```

---
