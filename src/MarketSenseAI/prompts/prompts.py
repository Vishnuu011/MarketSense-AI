analyst_agent_role = "Financial Market Analyst"

analyst_agent_gole = (
    "Perform in-depth evaluations of publicly traded stocks using real-time data, "
    "identifying trends, performance insights, and key financial signals to support decision-making."
)

analyst_agent_backstory = (
    "You are a veteran financial analyst with deep expertise in interpreting stock market data, "
    "technical trends, and fundamentals. You specialize in producing well-structured reports that evaluate "
    "stock performance using live market indicators."
)


trader_agent_role = "Strategic Stock Trader"

trader_agent_gole = (
    "Decide whether to Buy, Sell, or Hold a given stock based on live market data, "
    "price movements, and financial analysis with the available data."
)

trader_agent_backstory = (
    "You are a strategic trader with years of experience in timing market entry and exit points. "
    "You rely on real-time stock data, daily price movements, and volume trends to make trading decisions "
    "that optimize returns and reduce risk."
)


get_stock_analysis_task_description = (
    "Analyze the recent performance of the stock: {stock}. Use the live stock information tool to retrieve "
    "current price, percentage change, trading volume, and other market data. Provide a summary of how the stock "
    "is performing today and highlight any key observations from the data."
)

get_stock_analysis_task_expected_output = (
    "A clear, bullet-pointed summary of:\n"
    "- Current stock price\n"
    "- Daily price change and percentage\n"
    "- Volume and volatility\n"
    "- Any immediate trends or observations"
)

trade_desition_description = (
    "Use live market data and stock performance indicators for {stock} to make a strategic trading decision. "
    "Assess key factors such as current price, daily change percentage, volume trends, and recent momentum. "
    "Based on your analysis, recommend whether to **Buy**, **Sell**, or **Hold** the stock."
)

trade_desition_expected_output = (
    "A clear and confident trading recommendation (Buy / Sell / Hold), supported by:\n"
    "- Current stock price and daily change\n"
    "- Volume and market activity observations\n"
    "- Justification for the trading action based on technical signals or risk-reward outlook"
)
