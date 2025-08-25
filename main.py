from src.MarketSenseAI.agents.agent_and_task import AgentsAndTaskCrewOperation

if __name__ == "__main__":

    # model_path = "best_ddpg_agent.pkl"

    # train_or_load_agent(
    #     model_path=model_path,
    #     training_data=training_data,
    #     ticker="TATAMOTORS.NS"
    # )

    crew = AgentsAndTaskCrewOperation(
        temperature=0.7
    )
    print(crew.run_crew("TCS.NS"))

