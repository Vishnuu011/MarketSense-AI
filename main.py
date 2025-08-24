from src.MarketSenseAI.rl_model.rl_modeling import fetch_stock_data
from src.MarketSenseAI.rl_model.rl_modeling import train_or_load_agent, training_data, test_data, validation_data

if __name__ == "__main__":

    model_path = "best_ddpg_agent.pkl"

    train_or_load_agent(
        model_path=model_path,
        training_data=training_data,
        ticker="TATAMOTORS.NS"
    )