# 🛡️ Enterprise Ad-Policy RL Simulation

A custom, bleeding-edge Reinforcement Learning environment transformed into a robust, multi-service enterprise simulation to fulfill Scaler AI Labs requirements. This sandbox evaluates the ability of Vision-Language Models (VLMs) and LLMs to act as autonomous ad moderators, navigating complex policy violations, multimodal traps, and illegal targeting within a strict corporate workflow.

## 🚀 Core Features

- **Distributed Microservices Architecture**: Integrates with external REST APIs for Regulatory checks (Port 8001), CRM data (Port 8002), and Audit logging (Port 8003) to simulate a real enterprise environment.
- **Strict Workflow Phase-Gating**: Enforces enterprise compliance by requiring the AI agent to explicitly invoke `query_regulations`, `check_crm`, and `log_audit` before making any terminal decisions.
- **Reward Shaping**: Implements a strict -0.05 step penalty to force the AI agent to optimize tool usage and prevent infinite analysis loops, alongside severe penalties for workflow violations.
- **Multimodal Traps**: Tests the limits of VLMs by presenting ads where the text is benign, but the visual elements contain severe policy violations.
- **OpenEnv API**: Interacts via the `/step` endpoint for action processing and reward signaling.

## 📋 Evaluation Tasks

The environment natively supports 4 distinct adversarial tasks:

- `task_1_healthcare`: Evaluates ads for unapproved medical claims, pharmaceuticals, and subtle dog whistles.
- `task_2_financial`: Evaluates ads for predatory financial services, scams, and high-pressure tactics.
- `task_3_multimodal`: Detects policy violations hidden entirely within visual elements that bypass standard NLP text filters.
- `task_4_targeting`: Identifies illegal demographic targeting (e.g., adult financial services targeting minors).

## 🛠️ Available Agent Tools

The environment exposes the following action space to the evaluating LLM:

- `analyze_image`: Request VLM context for visual elements.
- `request_landing_page`: Extract simulated URL endpoints.
- `request_id_verification`: Check advertiser trust scores and targeting anomalies.
- `query_regulations`: Fetch regulatory policy summaries for the ad category.
- `check_crm`: Retrieve advertiser history and trust metrics from the CRM microservice.
- `log_audit`: Submit reasoning and actions to the audit ledger.
- `approve` / `reject`: Terminal actions.

## 🚦 Quick Start (Local)

1. **Build the Docker Image**
   ```bash
   docker build -t meta-ad-sandbox .
   ```

2. **Run the Environment Container**
   ```bash
   docker run -p 8000:8000 meta-ad-sandbox
   ```
   *(Note: Ensure the separate microservices on ports 8001, 8002, and 8003 are running).*

3. **Run the Automated Inference Agent**
   Make sure your Hugging Face credentials are set, then run the evaluation script:
   ```bash
   export HF_TOKEN="your_hugging_face_token"
   python inference.py
   ```