from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Compliance Audit API")

class AuditRecord(BaseModel):
    ad_id: str
    action_taken: str
    reasoning: str

@app.post("/log")
def log_audit(record: AuditRecord):
    return {"status": "success", "audit_id": f"AUD-{record.ad_id[:4]}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
