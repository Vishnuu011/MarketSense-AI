import os, sys

from src.MarketSenseAI.prompts.prompts import *
from src.MarketSenseAI.tools.tools import get_stock_price, predict_and_visualize_tool

from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from typing import Optional, Union, Any


class AgentsAndTaskCrewOperation:

    def __init__(
        self, 
        model: str = "groq/llama-3.3-70b-versatile", 
        temperature: Optional[Union[int, float]] = None
    ):

        try:
            self.groq_llm = ChatGroq(
                model=model,
                temperature=temperature
            )

        except Exception as e:
            print(f"[ERROR in groq llm model is not loaded]: {e}")


    def analyst_agent(self) -> Agent:

        return Agent(
            role=analyst_agent_role,
            goal=analyst_agent_gole,
            backstory=analyst_agent_backstory,
            tools=[get_stock_price],
            verbose=True,
            llm=self.groq_llm
        ) 

    def trader_agent(self) -> Agent:

        return Agent(
            role=trader_agent_role,
            goal=trader_agent_gole,
            backstory=trader_agent_backstory,
            tools=[],
            verbose=True,
            llm=self.groq_llm
        )   
    
    def get_stock_analysis_task(self) -> Task:

        return Task(
            description=get_stock_analysis_task_description,
            expected_output=get_stock_analysis_task_expected_output,
            agent=self.analyst_agent()
        )
    
    def trade_decision_task(self) -> Task:

        return Task(
            description=trade_desition_description,
            expected_output=trade_desition_expected_output,
            agent=self.trader_agent()
        )
    
    def agent_task_crew(self) -> Crew:

        return Crew(
            agents=[
                self.analyst_agent(),
                self.trader_agent()
            ],
            tasks=[
                self.get_stock_analysis_task(),
                self.trade_decision_task()
            ],
            verbose=True
        )
    
    def run_crew(self, stock: str) -> Any:

        try:
            crew = self.agent_task_crew()
            result = crew.kickoff(
                inputs={
                    "stock": stock
                }
            )

            return result
        
        except Exception as e:
            print(f"[ERROR in run_crew]: {e}")

