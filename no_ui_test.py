from config import config
from spnnier_ani import spinner
from colorama import Fore, Style, init
from src.MarketSenseAI.rl_model.rl_modeling import train_or_load_agent
from src.MarketSenseAI.rl_agent.ini_agent import init_rl_agent
from src.MarketSenseAI.rl_model.rl_modeling import add_technical_indicators, fetch_stock_data

# Initialize colorama for Windows compatibility
init(autoreset=True)

while True:
 
    symbol = input(Fore.CYAN + "Enter stock symbol (or type 'exit' to quit): ").strip()

    if symbol.lower() in ["exit", "quit", "q"]:
        spinner("exit")
        print(Fore.RED + "Exiting program... bye 👋")
        break

    config.symbol = symbol
    config.start_data = input(Fore.YELLOW + "Enter stock starting date (YYYY-MM-DD): ").strip()
    config.end_data = input(Fore.YELLOW + "Enter stock ending date (YYYY-MM-DD): ").strip()

    spinner("input")

    print(Fore.GREEN + f"\nSymbol set to: {config.symbol}")
    print(Fore.BLUE + f"Start date: {config.start_data}")
    print(Fore.MAGENTA + f"End date: {config.end_data}\n")

    stock_data = fetch_stock_data([config.symbol], config.start_data, config.end_data)
    full_training_data = add_technical_indicators(stock_data[config.symbol])

    print(full_training_data.head())
    print(full_training_data.shape)

  
    train_and_predict = input(Fore.GREEN + "Do you want to [train] or [skip]? ").strip()
    config.train_and_predict = train_and_predict

    if train_and_predict.lower() == "train":
        spinner("train_predict")
        model_path: str = "best_ddpg_agent.pkl"
        agent_rl = train_or_load_agent(
            model_path=model_path,
            training_data={config.symbol: full_training_data},
            ticker=config.symbol,
            total_timesteps=10000,
            threshold=0.1
        )
        print(Fore.GREEN + f"✅ Training complete for {config.symbol}")
    else:
        print(Fore.YELLOW + "⚠️ Skipping training...")
        continue

  
    predict_input = input(Fore.CYAN + "Do you want to [predict] or [skip]? ").strip()
    config.predict = predict_input

    if predict_input.lower() == "predict":
        spinner("train_predict")
        agent = init_rl_agent(model="llama-3.3-70b-versatile")
        result = agent.run(config.symbol)   # better to use symbol instead of user string
        print(Fore.GREEN + f"📊 Prediction result:\n{result}")
    else:
        print(Fore.YELLOW + "⚠️ Skipping prediction...")
        continue









    # # Display summary with colors
    # print(Fore.GREEN + f"\nNow using: {config.symbol}")
    # print(Fore.BLUE + f"Start date: {config.start_data}")
    # print(Fore.MAGENTA + f"End date: {config.end_data}\n")
