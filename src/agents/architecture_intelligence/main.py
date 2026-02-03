from src.agents.architecture_intelligence.agent import ArchitectureIntelligenceAgent

agent = ArchitectureIntelligenceAgent()
app = agent.app


@app.get("/")
async def root():
    return {
        "message": "Multi-Agent DevOps Agent is running!",
        "agent": agent.name,
        "available_tools": list(agent.tools.keys()),
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(agent.app, host="0.0.0.0", port=8305)
