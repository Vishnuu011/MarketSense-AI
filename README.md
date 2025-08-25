# 📈 MarketSense AI  

**MarketSense AI** is an interactive, CLI-based **stock trading framework** that combines:  
- 🤖 **Deep Reinforcement Learning (DDPG)** for autonomous trading decisions  
- 🧠 **LangChain + CrewAI multi-agent system** for natural language financial insights  
- 📊 **yFinance, Pandas, NumPy** for market data analysis  
- 📉 **Matplotlib** for visualization of portfolio performance and predictions  

The framework lets users **train RL agents, predict next-day stock actions, run AI-driven market analysis, and manage model artifacts** — all from an **interactive CLI**.  

---

## 🚀 Features  
- Interactive **CLI workflow** for trading and market analysis  
- Train **DDPG RL agent** on historical stock data (continuous-action trading)  
- Fetch and preprocess **real-time + historical** data via `yFinance`  
- Visualize predictions & portfolio performance with **Matplotlib**  
- Multi-agent reasoning with **LangChain + CrewAI**  
  - 📊 *StockTrader Agent* → RL-based action (Buy/Sell/Hold)  
  - 📰 *MarketAnalyst Agent* → Natural language market insights  
- Manage model artifacts (**train, load, delete pickle files**)  

---

## 🛠 Tech Stack  

- Python **3.11+**  
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) (DDPG)  
- [Gym](https://www.gymlibrary.dev/)  
- [yFinance](https://pypi.org/project/yfinance/)  
- [Pandas](https://pandas.pydata.org/) + [NumPy](https://numpy.org/)  
- [Matplotlib](https://matplotlib.org/)  
- [LangChain](https://www.langchain.com/)  
- [CrewAI](https://github.com/joaomdmoura/crewAI)  

---

## ⚙️ Installation  

```bash
# Clone the repository
git clone https://github.com/Vishnuu011/MarketSense-AI.git
cd MarketSense-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```
---

🔑 Setup .env

You need to create a .env file in the root folder for API keys and model configurations. Example:

```bash
API_KEY=your_api_key_here
MODEL_NAME=groq/llama-3.3-70b-versatile
```

(⚠️ Don’t commit .env to GitHub — keep it private!)

---

## ▶️ CLI Usage  

Run the CLI:  

```bash
python run_trading_agent.py
```

### Workflow  
1. Enter stock symbol + date range  
2. Train or load **DDPG agent**  
3. Predict next-day action (Buy/Sell/Hold)  
4. Run **CrewAI market analysis**  
5. Optionally remove `.pkl` files  


---

## 💻 Example CLI Interaction  

```bash
Enter stock symbol (or type 'exit' to quit): AAPL
Enter stock starting date (YYYY-MM-DD): 2023-01-01
Enter stock ending date (YYYY-MM-DD): 2023-06-30

Do you want to [train] or [skip]? train
✅ Training complete for AAPL

Do you want to [predict] or [skip]? predict
📊 Prediction result for next day: BUY

Do you want to [marketanalysis] or [skip]? marketanalysis
Stock market analysis result:
The stock shows strong bullish momentum with key technical support at $135.

Do you want [remove-pkl-file] or [skip]? remove-pkl-file
🗑 Pickle files have been removed.
```

---

## 📷 Screenshots  

### 🔹 Training & Prediction  
![Training and Prediction](assets/Screenshot%20(1).png)  

### 🔹 CrewAI Market Analysis  
![CrewAI Analysis](assets/Screenshot%20(2).png)  

### 🔹 Market Summary & Cleanup  
![Market Summary](assets/Screenshot%20(3).png)  

---

## 📊 Workflow Overview  

```mermaid
graph TD
    A[User Input CLI] --> B[Fetch Data (yFinance)]
    B --> C[Preprocess Data (Pandas, NumPy)]
    C --> D[Train DDPG Agent (Stable-Baselines3)]
    D --> E[Save/Load Model (Pickle)]
    E --> F[Predict Next-Day Action]
    F --> G[Visualization (Matplotlib)]
    G --> H[Market Analysis (LangChain + CrewAI)]
    H --> I[Final Decision: BUY / SELL / HOLD]

---
📊 Technical Indicators Used

- EMA (Exponential Moving Average)

- MACD (Moving Average Convergence Divergence)

- RSI (Relative Strength Index)

- CCI (Commodity Channel Index)

- ADX (+DI, -DI, DX)

---

## 📜 License  
This project is licensed under the **MIT License**.  

---

⚡ **MarketSense AI** → Bringing together **Reinforcement Learning + Multi-Agent LLMs** for smarter trading decisions.  
