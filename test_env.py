import requests
import json

BASE_URL = "http://localhost:8000"

def safe_post(endpoint, data=None):
    """Helper to catch errors before they crash the script."""
    try:
        url = f"{BASE_URL}/{endpoint}"
        response = requests.post(url, json=data)
        
        # If the server sent an error code (4xx or 5xx), print the text
        if response.status_code != 200:
            print(f"❌ Server Error {response.status_code}: {response.text}")
            return None
            
        return response.json()
    except Exception as e:
        print(f"⚠️ Request Failed: {e}")
        return None

def run_test():
    print("--- 🔄 Testing /reset ---")
    reset_data = safe_post("reset")
    if not reset_data: return
    
    obs = reset_data.get('observation', reset_data)
    print(f"Ad Loaded: {obs.get('headline', 'N/A')}\n")

    print("--- 🔍 Testing 'analyze_image' Tool ---")
    # Payload must be wrapped in 'action' for OpenEnv 2026
    step1_payload = {
        "action": {
            "action_type": "analyze_image",
            "reasoning": "Standard adversarial check."
        }
    }
    s1_data = safe_post("step", step1_payload)
    if s1_data:
        s1_obs = s1_data.get('observation', s1_data)
        print(f" {s1_obs.get('status_message', 'N/A')}\n")

    print("--- ⚖️ Testing 'query_regulations' Tool ---")
    reg_payload = {"action": {"action_type": "query_regulations", "reasoning": "Check policies"}}
    reg_data = safe_post("step", reg_payload)
    if reg_data:
        obs = reg_data.get('observation', reg_data)
        print(f" {obs.get('status_message', 'N/A')}\n")

    print("--- 🏢 Testing 'check_crm' Tool ---")
    crm_payload = {"action": {"action_type": "check_crm", "reasoning": "Check advertiser history"}}
    crm_data = safe_post("step", crm_payload)
    if crm_data:
        obs = crm_data.get('observation', crm_data)
        print(f" {obs.get('status_message', 'N/A')}\n")

    print("--- 📝 Testing 'log_audit' Tool ---")
    audit_payload = {"action": {"action_type": "log_audit", "reasoning": "Log action before reject"}}
    audit_data = safe_post("step", audit_payload)
    if audit_data:
        obs = audit_data.get('observation', audit_data)
        print(f" {obs.get('status_message', 'N/A')}\n")

    print("--- ✅ Testing Final Decision ---")
    step2_payload = {
        "action": {
            "action_type": "reject",
            "reasoning": "Detected policy violation."
        }
    }
    s2_data = safe_post("step", step2_payload)
    if s2_data:
        reward = s2_data.get('reward', 0.0)
        done = s2_data.get('done', s2_data.get('terminal', False))
        print(f"Final Reward: {reward} | Done: {done}")

if __name__ == "__main__":
    run_test()