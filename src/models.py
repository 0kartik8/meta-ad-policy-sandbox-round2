from typing import Literal, Optional, Dict, Any
from openenv.core.env_server import Action, Observation, State

class AdObservation(Observation):
    ad_id: str
    headline: str
    body_text: str
    advertiser_trust_score: float
    targeting_data: Dict[str, Any]
    image_url: str
    status_message: str
    
    # 🚨 NEW: OpenEnv requires these to be part of the Observation!
    reward: float = 0.0
    done: bool = False

class AdAction(Action):
    action_type: Literal[
        "approve", "reject", "analyze_image", 
        "request_landing_page", "request_id_verification",
        "query_regulations", "check_crm", "log_audit"
    ]
    reasoning: str
    violation_category: Optional[Literal["HEALTHCARE", "FINANCIAL", "NONE"]] = None

class AdState(State):
    step_count: int = 0
    total_reward: float = 0.0
    current_ad_id: Optional[str] = None