from fastapi import FastAPI
import uvicorn

app = FastAPI(title="Advertiser CRM API")

@app.get("/advertiser/{advertiser_id}")
def get_advertiser(advertiser_id: str):
    # Simulated database lookup
    return {
        "advertiser_id": advertiser_id,
        "prior_violations": 3 if "bad" in advertiser_id else 0,
        "summary": "High-risk offender" if "bad" in advertiser_id else "Clean record"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
