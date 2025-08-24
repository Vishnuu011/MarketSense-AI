import os, sys
import pandas as pd
import numpy as np
import yfinance as yf
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
import pickle
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings("ignore")


np.random.seed(42)


"""""
Download finance data fram yahoo finance API
"""

def fetch_stock_data(tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:

    try:
        stock_data: Dict[str, pd.DataFrame] = {}

        for ticker in tickers:
            
            df = yf.download(
                tickers=ticker,
                start=start_date,
                end=end_date
            )

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]

            df.dropna(inplace=True)

            if "Price" in df.columns:
                df = df.drop(
                    columns=["Price"]
                )  
            stock_data[ticker] = df

        return stock_data
              
    except Exception as e:
        print(f"ERROR in fetch_stock_data: {e}")

stock_data = fetch_stock_data(
    tickers=["TATAMOTORS.NS"],
    start_date="2009-01-01",
    end_date="2025-08-23"
)        

print(stock_data['TATAMOTORS.NS'].head())      

training_data_time_range = ('2009-01-01', '2015-12-31')
validation_data_time_range = ('2016-01-01', '2016-12-31')
test_data_time_range = ('2017-01-01', '2025-08-22')

training_data = {}
validation_data = {}
test_data = {}


for ticker, df in stock_data.items():
    training_data[ticker] = df.loc[training_data_time_range[0]:training_data_time_range[1]]
    validation_data[ticker] = df.loc[validation_data_time_range[0]:validation_data_time_range[1]]
    test_data[ticker] = df.loc[test_data_time_range[0]:test_data_time_range[1]]


for ticker, df in stock_data.items():
    print(f'- Training data shape for {ticker}: {training_data[ticker].shape}')
    print(f'- Validation data shape for {ticker}: {validation_data[ticker].shape}')
    print(f'- Test data shape for {ticker}: {test_data[ticker].shape}\n')
    


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:

    try:
        df = df.copy()

        # Calculate EMA 12 and 26 for MACD
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Calculate RSI 14
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Calculate CCI 20
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = tp.rolling(window=20).mean()
        mean_dev = tp.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df['CCI'] = (tp - sma_tp) / (0.015 * mean_dev)

        # Calculate ADX 14
        high_diff = df['High'].diff()
        low_diff = df['Low'].diff()
        df['+DM'] = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        df['-DM'] = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

        tr = pd.concat([
            df['High'] - df['Low'],
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        ], axis=1).max(axis=1)

        atr = tr.ewm(span=14, adjust=False).mean()
        df['+DI'] = 100 * (df['+DM'].ewm(span=14, adjust=False).mean() / atr)
        df['-DI'] = 100 * (df['-DM'].ewm(span=14, adjust=False).mean() / atr)
        dx = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
        df['ADX'] = dx.ewm(span=14, adjust=False).mean()

        df.dropna(inplace=True)
        return df

    except Exception as e:
        print(f"[ERROR in add_technical_indicators]: {e}")


# Add technical indicators
for ticker, df in training_data.items():
    training_data[ticker] = add_technical_indicators(df)
for ticker, df in validation_data.items():
    validation_data[ticker] = add_technical_indicators(df)
for ticker, df in test_data.items():
    test_data[ticker] = add_technical_indicators(df)



class StockTradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000) -> None:
        try:
            super(StockTradingEnv, self).__init__()
            self.df: pd.DataFrame = df.reset_index(drop=True)
            self.initial_balance: float = initial_balance
            self.current_step: int = 0

            # Action space: continuous value [-1, 1]
            self.action_space: spaces.Box = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )

            # Observation space: features + balance + holdings
            self.observation_space: spaces.Box = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(len(self.df.columns) + 2,),
                dtype=np.float32,
            )
        except Exception as e:
            print(f"[ERROR in __init__]: {e}")
            raise

    def reset(
        self, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        try:
            super().reset(seed=seed)
            self.balance: float = self.initial_balance
            self.net_worth: float = self.initial_balance
            self.holdings: int = 0
            self.current_step: int = 0
            return self._get_observation(), {}
        except Exception as e:
            print(f"[ERROR in reset]: {e}")
            raise

    def _get_observation(self) -> np.ndarray:
        try:
            row = self.df.iloc[self.current_step]
            return np.array([*row.values, self.balance, self.holdings], dtype=np.float32)
        except Exception as e:
            print(f"[ERROR in _get_observation]: {e}")
            raise

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        try:
            if self.current_step >= len(self.df) - 1:
                return self._get_observation(), 0.0, True, False, {}

            row = self.df.iloc[self.current_step]
            close_price: float = float(row["Close"])
            action_value: float = float(action[0])

            # Buy
            if action_value > 0:
                max_shares_to_buy: int = int(self.balance // close_price)
                shares_to_buy: int = int(max_shares_to_buy * action_value)
                if shares_to_buy > 0 and self.balance >= shares_to_buy * close_price:
                    self.holdings += shares_to_buy
                    self.balance -= shares_to_buy * close_price

            # Sell
            elif action_value < 0:
                shares_to_sell: int = int(self.holdings * abs(action_value))
                if shares_to_sell > 0:
                    self.holdings -= shares_to_sell
                    self.balance += shares_to_sell * close_price

            self.net_worth = self.balance + self.holdings * close_price
            reward: float = self.net_worth - self.initial_balance
            self.current_step += 1
            terminated: bool = self.current_step >= len(self.df) - 1

            return self._get_observation(), reward, terminated, False, {}
        except Exception as e:
            print(f"[ERROR in step at step={self.current_step}]: {e}")
            raise


class DDPGAgent:
    def __init__(
        self,
        env: Optional[Any] = None,
        total_timesteps: Optional[int] = None,
        threshold: float = 0.1,
    ) -> None:
       
        self.model: Optional[DDPG] = None
        self.threshold: float = threshold

        if env is not None and total_timesteps is not None:
            self.train(env, total_timesteps)

    def train(self, env: Any, total_timesteps: int) -> None:
        try:
            self.model = DDPG("MlpPolicy", env, verbose=1, device="cpu")
            self.model.learn(total_timesteps=total_timesteps)
        except Exception as e:
            print(f"[ERROR in train]: {e}")
            raise

    def predict(self, obs: np.ndarray) -> np.ndarray:
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded. Call load() first.")
            action, _ = self.model.predict(obs, deterministic=True)
            return action
        except Exception as e:
            print(f"[ERROR in predict]: {e}")
            raise

    def action_to_recommendation(self, action: np.ndarray) -> str:
        try:
            action_value: float = float(action[0])
            if action_value > self.threshold:
                return "BUY"
            elif action_value < -self.threshold:
                return "SELL"
            else:
                return "HOLD"
        except Exception as e:
            print(f"[ERROR in action_to_recommendation]: {e}")
            raise

    def save(self, file_path: str) -> None:
        try:
            if self.model is None:
                raise RuntimeError("No model to save.")
            # Save SB3 model
            self.model.save(file_path)
            # Save threshold separately
            with open(file_path + "_meta.pkl", "wb") as f:
                pickle.dump({"threshold": self.threshold}, f)
        except Exception as e:
            print(f"[ERROR in save]: {e}")
            raise

    @classmethod
    def load(cls, file_path: str) -> "DDPGAgent":
        try:
            # Load SB3 model
            model: DDPG = DDPG.load(file_path, device="cpu")

            # Load threshold metadata
            meta_path = file_path + "_meta.pkl"
            if os.path.exists(meta_path):
                with open(meta_path, "rb") as f:
                    meta = pickle.load(f)
                threshold = meta.get("threshold", 0.1)
            else:
                threshold = 0.1

            agent = cls(env=None, total_timesteps=None, threshold=threshold)
            agent.model = model
            return agent
        except Exception as e:
            print(f"[ERROR in load]: {e}")
            raise


def prepare_next_day_prediction_data(
    ticker: str = "TATAMOTORS.NS", lookback: int = 100
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    try:
        # Download 6 months of daily data
        df = yf.download(ticker, period="6mo", interval="1d")

        # Handle multi-indexed columns (common with yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        # Add indicators + clean
        df.dropna(inplace=True)
        df = add_technical_indicators(df)
        df = df.dropna()

        if df.empty:
            raise ValueError(f"No usable data available for {ticker}")

        # Most recent sample
        latest_data: pd.DataFrame = df.tail(1).copy()

        # Historical context
        historical_data: pd.DataFrame = df.tail(lookback).copy()

        return latest_data, historical_data

    except Exception as e:
        print(f"[ERROR in prepare_next_day_prediction_data for {ticker}]: {e}")
        raise        


def predict_next_day(model_path: str, ticker: str = "TATAMOTORS.NS") -> Dict[str, Any]:
   
    try:
        # Load trained agent
        agent = DDPGAgent.load(model_path)

        # Get latest + historical data
        latest_data, historical_data = prepare_next_day_prediction_data(ticker)

        # Build single-step env for prediction
        env = StockTradingEnv(latest_data)
        obs, _ = env.reset()

        # Predict action
        action = agent.predict(obs)
        recommendation = agent.action_to_recommendation(action)

        # Extract features for reporting
        last_row = latest_data.iloc[-1]
        current_price: float = float(last_row["Close"])

        technicals: Dict[str, float] = {
            "MACD": float(last_row["MACD"]),
            "RSI": float(last_row["RSI"]),
            "CCI": float(last_row["CCI"]),
            "ADX": float(last_row["ADX"]),
        }

        return {
            "ticker": ticker,
            "date": latest_data.index[-1].strftime("%Y-%m-%d"),
            "recommendation": recommendation,
            "action_value": float(action[0]),
            "current_price": current_price,
            "technical_indicators": technicals,
            "historical_data": historical_data,  # keep full context for visualization
        }

    except Exception as e:
        print(f"[ERROR in predict_next_day for {ticker}]: {e}")
        raise


def visualize_prediction(prediction: Dict[str, Any], historical_data: pd.DataFrame, output_dir: str = "outputs") -> str:
   
    try:
        ticker = prediction.get("ticker", "Unknown")
        date = prediction.get("date", "N/A")
        action_value: float = prediction.get("action_value", 0.0)

        plt.figure(figsize=(15, 10))

        # Price chart
        plt.subplot(2, 2, 1)
        plt.plot(historical_data.index[-50:], historical_data["Close"][-50:], "b-")
        plt.title(f"{ticker} Price (Last 50 Days)")
        plt.ylabel("Price")
        plt.grid(True)

        # MACD
        plt.subplot(2, 2, 2)
        plt.plot(historical_data.index[-50:], historical_data["MACD"][-50:], "orange", label="MACD")
        plt.plot(historical_data.index[-50:], historical_data["Signal"][-50:], "red", label="Signal")
        plt.title("MACD Indicator")
        plt.legend()
        plt.grid(True)

        # RSI
        plt.subplot(2, 2, 3)
        plt.plot(historical_data.index[-50:], historical_data["RSI"][-50:], "purple")
        plt.axhline(70, color="red", linestyle="--", alpha=0.7)
        plt.axhline(30, color="green", linestyle="--", alpha=0.7)
        plt.title("RSI Indicator")
        plt.ylabel("RSI")
        plt.grid(True)

        # Prediction visualization
        plt.subplot(2, 2, 4)
        colors = ["red", "gray", "green"]
        recommendations = ["SELL", "HOLD", "BUY"]
        idx = 0 if action_value < -0.1 else (1 if abs(action_value) <= 0.1 else 2)

        plt.bar(["Recommendation"], [1], color=colors[idx])
        plt.text(
            0, 0.5, recommendations[idx],
            ha="center", va="center",
            fontsize=20, fontweight="bold"
        )
        plt.title(f"Prediction for {ticker} on {date}\nAction Value: {action_value:.3f}")
        plt.ylim(0, 1)
        plt.axis("off")

        plt.tight_layout()

        from datetime import datetime
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{ticker}_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close()

        return filepath

    except Exception as e:
        print(f"[ERROR in visualize_prediction]: {e}")
        raise    


def train_or_load_agent(model_path, training_data, ticker, total_timesteps=10000, threshold=0.1):
    
    try:
        if not os.path.exists(model_path):
            print("Training new DDPG Agent...")
            
            # Create training environment
            train_env = DummyVecEnv([lambda: StockTradingEnv(training_data[ticker])])

            # Initialize and train agent
            agent = DDPGAgent(train_env, total_timesteps=total_timesteps, threshold=threshold)

            # Save agent
            agent.save(model_path)
            print(f"✅ Model trained and saved to {model_path}")
        else:
            print(f"📂 Loading existing model from {model_path}")
            agent = DDPGAgent.load(model_path)

        return agent
    
    except Exception as e:
        print(f"❌ Error in training/loading agent: {e}")
        return None    