from config import config
import os,sys
from spnnier_ani import spinner
from colorama import Fore, Style, init
from src.MarketSenseAI.rl_model.rl_modeling import train_or_load_agent
from src.MarketSenseAI.rl_agent.ini_agent import init_rl_agent
from src.MarketSenseAI.rl_model.rl_modeling import add_technical_indicators, fetch_stock_data
from src.MarketSenseAI.agents.agent_and_task import AgentsAndTaskCrewOperation
from PIL import Image
init(autoreset=True)


output_dir = "outputs"  

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

    print(Fore.YELLOW + str(full_training_data.head()))
    print(Fore.CYAN + str(full_training_data.shape))

   
    train_and_predict = input(Fore.GREEN + "Do you want to [train] or [skip]? ").strip().lower()
    config.train_and_predict = train_and_predict

    if train_and_predict == "train":
        spinner("train_predict")
        model_path = "best_ddpg_agent.pkl"
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

    
    predict_input = input(Fore.CYAN + "Do you want to [predict] or [skip]? ").strip().lower()
    config.predict = predict_input

    if predict_input == "predict":
        spinner("train_predict")
        agent = init_rl_agent(model="llama-3.3-70b-versatile")
        result = agent.run(config.symbol)
        print(Fore.GREEN + f"📊 Prediction result for next day:\n{result}")

        if os.path.exists(output_dir):
            image_files = [f for f in os.listdir(output_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
            for img_file in image_files:
                img_path = os.path.join(output_dir, img_file)
                try:
                    img = Image.open(img_path)
                    img.show()
                    input(Fore.CYAN + f"Press Enter after closing {img_file}...")
                except Exception as e:
                    print(Fore.RED + f"⚠️ Could not open {img_file}: {e}")
                finally:
                    os.remove(img_path)

    else:
        print(Fore.YELLOW + "⚠️ Skipping prediction...")

    
    market_analyst_input = input(Fore.MAGENTA + "Do you want to [marketanalysis] or [skip]? ").strip().lower()
    config.market_analyst = market_analyst_input

    if market_analyst_input == "marketanalysis":
        spinner("market_analysis")
        stock_market_agent = AgentsAndTaskCrewOperation(
            model="groq/llama-3.3-70b-versatile",
            temperature=0.7
        )
        result_m = stock_market_agent.run_crew(stock=config.symbol)
        print(Fore.MAGENTA + f"Stock market analysis result:\n{result_m}")
    else:
        print(Fore.YELLOW + "⚠️ Skipping stock market analysis...")

    
    remove_pkl_file = input(Fore.RED + "Do you want [remove-pkl-file] or [skip]? ").strip().lower()
    config.remove_pkl_file = remove_pkl_file

    if remove_pkl_file == "remove-pkl-file":
        spinner("delete_model")
        for f in ["best_ddpg_agent.pkl", "best_ddpg_agent.pkl_meta.pkl"]:
            if os.path.exists(f):
                os.remove(f)
        print(Fore.RED + f"Pickle files have been removed.")
    else:
        print(Fore.YELLOW + "⚠️ Skipping remove pickle file operation ...")
    
