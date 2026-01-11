from src.agents.image.agent import ArchitectureIntelligenceAgent

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
