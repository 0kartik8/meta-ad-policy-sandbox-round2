import uvicorn
from openenv.core.env_server import create_fastapi_app
from src.environment import AdPolicyEnvironment
from src.models import AdAction, AdObservation

# 1. Create the App
# NOTICE: We pass the CLASS NAME (AdPolicyEnvironment), not 'env' or 'AdPolicyEnvironment()'
app = create_fastapi_app(
    AdPolicyEnvironment, 
    AdAction, 
    AdObservation
)

def main():
    print("🚀 Starting Meta Ad-Policy Sandbox on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
