import os
import json
import requests
from openai import OpenAI

# ── MANDATORY VARIABLES ────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy_local_token")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-70B-Instruct")

ENV_URL   = "http://localhost:8000"
MAX_STEPS = 12  # 3 required tools + buffer for retries

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

TASKS = [
    "task_1_healthcare",
    "task_2_financial",
    "task_3_multimodal",
    "task_4_targeting",
]

# ── STRICT GRADING LOGGERS ─────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ── SYSTEM PROMPT ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an enterprise Meta Ad-Policy Compliance Agent.
Your job: review ads and make approve/reject decisions.

MANDATORY WORKFLOW (in order, every episode):
  1. query_regulations   — fetch policy for this ad category
  2. check_crm           — verify advertiser history
  3. analyze_image       — REQUIRED if task is multimodal or image looks suspicious
  4. request_id_verification — use if targeting_data age_min looks low (< 18)
  5. log_audit           — ALWAYS submit audit before final decision
  6. approve / reject    — terminal action

COST: each tool call costs -0.05. Skip unnecessary tools.
PENALTY: skipping query_regulations, check_crm, or log_audit before deciding = -0.5.
REWARD: correct final decision = +1.0 | wrong = -1.0

VIOLATION SIGNALS:
- HEALTHCARE: prescription claims, unapproved cures, before/after weight photos, GLP-1 analogs
- FINANCIAL: guaranteed returns, crypto scams, minor targeting (age_min < 18), predatory loans
- MULTIMODAL: text looks clean but image contains the violation — always analyze_image on task_3

Respond ONLY with valid JSON:
{"action_type": "<action>", "reasoning": "<brief reason>", "violation_category": "HEALTHCARE|FINANCIAL|NONE"}
"""

# ── LLM CALL ───────────────────────────────────────────────────────────────────
def get_llm_action(history: list, observation: dict) -> dict:
    history.append({
        "role": "user",
        "content": f"Current observation:\n{json.dumps(observation, indent=2)}\n\nWhat is your next action?"
    })
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=256,
        )
        reply = response.choices[0].message.content.strip()
        history.append({"role": "assistant", "content": reply})

        # Strip markdown fences if present
        if reply.startswith("```"):
            reply = reply.split("```")[1]
            if reply.startswith("json"):
                reply = reply[4:]
            reply = reply.strip()

        result = json.loads(reply)
        return {
            "action_type":        result.get("action_type", "query_regulations"),
            "reasoning":          result.get("reasoning", "Fallback"),
            "violation_category": result.get("violation_category", "NONE"),
        }
    except Exception as e:
        # Safe fallback — start the required workflow
        fallback = {"action_type": "query_regulations", "reasoning": f"Error recovery: {e}", "violation_category": "NONE"}
        history.append({"role": "assistant", "content": json.dumps(fallback)})
        return fallback

# ── MAIN ───────────────────────────────────────────────────────────────────────
def main() -> None:
    for task_id in TASKS:
        log_start(task=task_id, env="meta_ad_policy_sandbox", model=MODEL_NAME)

        rewards     = []
        steps_taken = 0
        success     = False
        history     = []   # Per-task conversation memory

        try:
            # 1. Reset environment
            res = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=10)
            if res.status_code != 200:
                log_step(step=1, action="reset_failed", reward=0.0, done=True, error=f"HTTP {res.status_code}")
                log_end(success=False, steps=0, score=0.01, rewards=[])
                continue

            # Observation lives directly in the response (reward + done embedded)
            observation = res.json()
            done = observation.get("done", False)

            # 2. Agent loop
            while not done and steps_taken < MAX_STEPS:
                steps_taken += 1

                action_payload = get_llm_action(history, observation)
                action_str     = action_payload["action_type"]

                # Send action flat — matches AdAction Pydantic model
                step_res = requests.post(f"{ENV_URL}/step", json=action_payload, timeout=10)

                if step_res.status_code != 200:
                    log_step(step=steps_taken, action=action_str, reward=0.0, done=True,
                             error=f"HTTP {step_res.status_code}")
                    break

                step_data = step_res.json()

                # Reward and done are inside the observation object
                reward = step_data.get("reward", 0.0)
                done   = step_data.get("done", False)
                observation = step_data   # Full obs passed back next iteration

                rewards.append(reward)
                log_step(step=steps_taken, action=action_str, reward=reward, done=done)

            # 3. Final score
            raw_score = sum(rewards)
            success   = raw_score > 0.5
            log_end(success=success, steps=steps_taken, score=raw_score, rewards=rewards)

        except Exception as e:
            err = str(e).replace("\n", " ")
            log_step(step=steps_taken + 1, action="exception", reward=0.0, done=True, error=err)
            log_end(success=False, steps=steps_taken, score=0.01, rewards=rewards)

if __name__ == "__main__":
    main()  
