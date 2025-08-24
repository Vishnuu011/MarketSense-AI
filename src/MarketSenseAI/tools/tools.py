import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.MarketSenseAI.rl_model.rl_modeling import (
    prepare_next_day_prediction_data,
    predict_next_day,
    visualize_prediction
)

import warnings
from langchain.agents.initialize import initialize_agent, AgentType
from langchain_groq import ChatGroq
from langchain.tools import StructuredTool
import yfinance as yf
from crewai.tools import tool
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")


def predict_and_visualize_func(model_path: str="best_ddpg_agent.pkl", ticker: str = "TATAMOTORS.NS") -> str:

    try:
        prediction = predict_next_day(model_path, ticker)

        lines = []
        lines.append("="*50)
        lines.append("PREDICTION RESULTS")
        lines.append("="*50)
        lines.append(f"Ticker: {prediction['ticker']}")
        lines.append(f"Date: {prediction['date']}")
        lines.append(f"Recommendation: {prediction['recommendation']}")
        lines.append(f"Action Value: {prediction['action_value']:.3f}")
        lines.append(f"Current Price: {prediction['current_price']:.2f}")
        lines.append("")

        tech = prediction['technical_indicators']
        lines.append("Technical Indicators:")
        lines.append(f"  MACD: {tech['MACD']:.4f}")
        lines.append(f"  RSI: {tech['RSI']:.2f}")
        lines.append(f"  CCI: {tech['CCI']:.2f}")
        lines.append(f"  ADX: {tech['ADX']:.2f}")
        lines.append("")

        lines.append("Technical Analysis:")
        if tech['RSI'] > 70:
            lines.append("  - RSI indicates OVERBOUGHT conditions")
        elif tech['RSI'] < 30:
            lines.append("  - RSI indicates OVERSOLD conditions")

        if tech['MACD'] > 0:
            lines.append("  - MACD indicates BULLISH momentum")
        else:
            lines.append("  - MACD indicates BEARISH momentum")

        if tech['ADX'] > 25:
            lines.append("  - ADX indicates STRONG TREND")

        lines.append("")
        lines.append("Prediction completed successfully!")

        # Visualization (local only)
        _, historical_data = prepare_next_day_prediction_data(ticker)
        visualize_prediction(prediction, historical_data)

        return "\n".join(lines)
    
    except Exception as e:
        print(f"[ERROR in predict_and_visualize_func]: {e}")



predict_and_visualize_tool = StructuredTool.from_function(
    func=predict_and_visualize_func,
    name="predict_and_visualize",
    description="Predict next-day stock action, return analysis and show visualization."
)

#""""test"""""

# print(predict_and_visualize_tool.run("best_ddpg_agent.pkl","TATAMOTORS.NS"))
# llm = ChatGroq(
#     model = "llama-3.3-70b-versatile",
#     temperature=0.5
# )

# agent = initialize_agent(
#     tools=[predict_and_visualize_tool],
#     llm=llm,
#     agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True
# )

# agent.run("Predict tomorrow's stock action for ticker TATAMOTORS.NS")
#"""""test"""

@tool("Live Stock Information Tool")
def get_stock_price(stock_symbol: str) -> str:
  
    try:
        stock = yf.Ticker(stock_symbol)
        info = stock.info

        current_price = info.get("regularMarketPrice")
        change = info.get("regularMarketChange")
        change_percent = info.get("regularMarketChangePercent")
        currency = info.get("currency", "USD")

        if current_price is None:
            return f"Could not fetch price for {stock_symbol}. Please check the symbol."

        return (
            f"Stock: {stock_symbol.upper()}\n"
            f"Price: {current_price} {currency}\n"
            f"Change: {change} ({round(change_percent, 2)}%)"
        )
    
    except Exception as e:
        print(f"[ERROR get_stock_price (tool)]: {e}")

