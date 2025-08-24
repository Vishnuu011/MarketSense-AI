from src.MarketSenseAI.rl_model.rl_modeling import prepare_next_day_prediction_data, predict_next_day, visualize_prediction


if __name__ == "__main__":
    model_path =r"C:\Users\VISHNU\Desktop\stock_trade_agent\MarketSense-AI\best_ddpg_agent.pkl"

    # Make prediction for next day
    print("\nMaking prediction for 2025-08-23...")
    prediction = predict_next_day(model_path)

    # Display results
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print(f"Ticker: {prediction['ticker']}")
    print(f"Date: {prediction['date']}")
    print(f"Recommendation: {prediction['recommendation']}")
    print(f"Action Value: {prediction['action_value']:.3f}")

    print(f"Current Price: ${prediction['current_price']:.2f}")

    print("\nTechnical Indicators:")
    tech = prediction['technical_indicators']
    print(f"  MACD: {tech['MACD']:.4f}")
    print(f"  RSI: {tech['RSI']:.2f}")
    print(f"  CCI: {tech['CCI']:.2f}")
    print(f"  ADX: {tech['ADX']:.2f}")

    # Interpret the signals
    print("\nTechnical Analysis:")
    if tech['RSI'] > 70:
        print("  - RSI indicates OVERBOUGHT conditions")
    elif tech['RSI'] < 30:
        print("  - RSI indicates OVERSOLD conditions")

    if tech['MACD'] > 0:
        print("  - MACD indicates BULLISH momentum")
    else:
        print("  - MACD indicates BEARISH momentum")

    if tech['ADX'] > 25:
        print("  - ADX indicates STRONG TREND")

    # Visualize the prediction
    _, historical_data = prepare_next_day_prediction_data()
    visualize_prediction(prediction, historical_data)

    print("\nPrediction completed successfully!")