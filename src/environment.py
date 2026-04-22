import uuid
from openenv.core.env_server import Environment
from src.models import AdAction, AdObservation, AdState
from src.generator import AdGenerator

# Module-level singleton session state.
# OpenEnv HTTP mode creates a new env instance per request and destroys it.
# We persist cross-request flags here, keyed by ad_id.
import threading

_SESSION_LOCK = threading.Lock()
_SESSION: dict = {
    "ad": None,
    "image_analyzed": False,
    "regulations_queried": False,
    "crm_checked": False,
    "audit_submitted": False,
    "step_count": 0,
    "total_reward": 0.0,
}

class AdPolicyEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.generator = AdGenerator()
        # Mirror module-level session into instance attrs for convenience
        self._sync_from_session()

    def _sync_from_session(self):
        """Pull latest session state into instance attributes."""
        with _SESSION_LOCK:
            self.current_ad = _SESSION["ad"]
            self.image_analyzed = _SESSION["image_analyzed"]
            self.regulations_queried = _SESSION["regulations_queried"]
            self.crm_checked = _SESSION["crm_checked"]
            self.audit_submitted = _SESSION["audit_submitted"]
            self.step_count = _SESSION["step_count"]
            self.total_reward = _SESSION["total_reward"]

    def _push_to_session(self):
        """Push instance state back into the module-level store."""
        with _SESSION_LOCK:
            _SESSION["ad"] = self.current_ad
            _SESSION["image_analyzed"] = self.image_analyzed
            _SESSION["regulations_queried"] = self.regulations_queried
            _SESSION["crm_checked"] = self.crm_checked
            _SESSION["audit_submitted"] = self.audit_submitted
            _SESSION["step_count"] = self.step_count
            _SESSION["total_reward"] = self.total_reward

    def _ensure_ad(self):
        self._sync_from_session()
        if self.current_ad is None:
            self.current_ad = self.generator.generate_random_ad()
            self.current_ad["task_id"] = self.current_ad.get("task_id") or ""
            self._push_to_session()

    def state(self) -> AdState:
        self._sync_from_session()
        ad_id = self.current_ad.get("ad_id") if self.current_ad else None
        return AdState(
            step_count=self.step_count,
            total_reward=self.total_reward,
            current_ad_id=ad_id
        )

    # Add task_id as an optional parameter
    def reset(self, task_id: str = None) -> AdObservation:
        # Pass the task_id down to the generator
        self.current_ad = self.generator.generate_random_ad(task_id)
        self.current_ad["task_id"] = task_id or ""
        self.image_analyzed = False
        self.step_count = 0
        self.total_reward = 0.0
        self.regulations_queried = False
        self.crm_checked = False
        self.audit_submitted = False
        self._push_to_session()
        
        # Add the task_id to the welcome message so the bot knows it worked
        msg = f"Ad loaded for {task_id}. Awaiting review." if task_id else "Random ad loaded. Awaiting review."
        return self._get_obs(msg)

    def step(self, action: AdAction) -> AdObservation: 
        self._sync_from_session()
        self._ensure_ad()
        self.step_count += 1
        
        reward = 0.0
        done = False
        message = "Action processed."

        if not action or not hasattr(action, 'action_type'):
            # Heavy penalty for invalid formatting to train the agent faster
            reward = -0.1 
            self.total_reward += reward
            self._push_to_session()
            return self._get_obs("Invalid action.", reward, False)

        act_type = str(action.action_type).lower()

        # 🧠 REWARD SHAPING: Tool Usage Penalties
        # We charge the agent -0.05 for using tools to force efficiency
        if act_type in ["analyze_image", "request_landing_page", "request_id_verification", "query_regulations", "check_crm", "log_audit"]:
            reward = -0.05
            
            if act_type == "analyze_image":
                self.image_analyzed = True
                vlm_text = self.current_ad.get('vlm_desc', 'No visual description.')
                message = vlm_text  # Cleaned up the double "VLM Output:" prefix here!
            
            elif act_type == "request_landing_page":
                # Return ambiguous category info, NOT ground truth
                category = self.current_ad.get("category", "unknown")
                domain_age = self.current_ad.get("domain_age_days", 365)
                message = f"Domain age: {domain_age} days. Category: {category}."
                
            elif act_type == "request_id_verification":
                targeting = self.current_ad.get("targeting_data", {})
                # NEW — task_id check removed, fires on any minor-targeting ad
                age_min = targeting.get("age_min", 18)
                if age_min < 18:
                    message = f"Warning: Campaign targets age {age_min}+. Financial product detected."
                else:
                    message = "Advertiser profile verified. No age-targeting anomalies."
            
            elif act_type == "query_regulations":
                import requests
                self.regulations_queried = True
                category = self.current_ad.get("category", "general")
                try:
                    resp = requests.get(f"http://localhost:8001/regulations/{category}")
                    message = resp.json().get("policy_summary", "Regulatory info unavailable.")
                except Exception:
                    message = "Failed to query regulations. Server offline."

            elif act_type == "check_crm":
                import requests
                self.crm_checked = True
                ad_id = self.current_ad.get("ad_id", "unknown")
                advertiser_id = self.current_ad.get("advertiser_id", ad_id)
                try:
                    resp = requests.get(f"http://localhost:8002/advertiser/{advertiser_id}")
                    message = resp.json().get("summary", "CRM info unavailable.")
                except Exception:
                    message = "Failed to check CRM. Server offline."
                    
            elif act_type == "log_audit":
                import requests
                self.audit_submitted = True
                ad_id = self.current_ad.get("ad_id", "unknown")
                payload = {"ad_id": ad_id, "action_taken": "audit", "reasoning": getattr(action, "reasoning", "none")}
                try:
                    resp = requests.post("http://localhost:8003/log", json=payload)
                    message = f"Audit logged successfully. ID: {resp.json().get('audit_id')}"
                except Exception:
                    message = "Failed to log audit. Server offline."

        # 🎯 FINAL DECISION: Big Rewards / Big Penalties
        elif act_type in ["approve", "reject"]:
            if not self.regulations_queried or not self.crm_checked or not self.audit_submitted:
                reward = -0.5
                self.total_reward += reward
                self._push_to_session()
                return self._get_obs(
                    "Action Denied: Enterprise workflow violation. You must query_regulations, check_crm, and log_audit before making a final decision.",
                    reward, False
                )

            task = self.current_ad.get("task_id", "")
            if "multimodal" in task and not self.image_analyzed:
                reward = -0.3
                self.total_reward += reward
                self._push_to_session()
                return self._get_obs(
                    "Cannot decide without image analysis on visual content review tasks.",
                    reward, False  # NOT done — agent must try again
                )
            done = True
            is_violation = self.current_ad.get("ground_truth", False)
            is_correct = (act_type == "reject" and is_violation) or \
                         (act_type == "approve" and not is_violation)
            
            reward = 1.0 if is_correct else -1.0
            message = f"Decision: {act_type.upper()}. " + ("Correct!" if is_correct else "Incorrect.")

        # Accumulate the total reward for the episode state
        self.total_reward += reward
        self._push_to_session()
        return self._get_obs(message, reward, done)

    def _get_obs(self, message: str, reward: float = 0.0, done: bool = False) -> AdObservation:
        self._ensure_ad()
        return AdObservation(
            ad_id=str(self.current_ad.get("ad_id", "N/A")),
            headline=str(self.current_ad.get("headline", "N/A")),
            body_text=str(self.current_ad.get("body_text", "N/A")),
            advertiser_trust_score=float(self.current_ad.get("advertiser_trust_score", 0.0)),
            targeting_data=dict(self.current_ad.get("targeting_data", {})),
            image_url=str(self.current_ad.get("image_url", "N/A")),
            status_message=str(message),
            reward=reward, 
            done=done       
        )