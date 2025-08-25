import os,sys
from dotenv import load_dotenv

from src.MarketSenseAI.tools.tools import predict_and_visualize_tool

from langchain.agents.initialize import initialize_agent, AgentType
from langchain.agents import AgentExecutor

from langchain_groq import ChatGroq
from typing import Optional

load_dotenv()


def init_rl_agent(model: str) -> Optional[AgentExecutor]:

    try:
        llm = ChatGroq(
            model=model,
            temperature=0.7
        )

        agent = initialize_agent(
            tools=[predict_and_visualize_tool],
            llm=llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        
        return agent

    except Exception as e:
        print(f"[ERROR in init_rl_agent]: {e}")