from fastapi import FastAPI
import uvicorn

app = FastAPI(title="Regulatory DB API")

REGULATIONS = {
    "healthcare": {
        "policy_summary": "Health claims require FDA approval. Prohibited: unverified cure claims, 'guaranteed results'.",
        "risk_level": "high"
    },
    "financial": {
        "policy_summary": "Financial ads require SEC registration. Prohibited: predatory APR above 36%.",
        "risk_level": "high"
    },
    "general": {
        "policy_summary": "Standard advertising standards apply. No deceptive claims.",
        "risk_level": "low"
    }
}

@app.get("/regulations/{category}")
def get_regulations(category: str):
    return REGULATIONS.get(category.lower(), REGULATIONS["general"])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
