import random
import uuid

class AdGenerator:
    # 1. Obvious Violations (Easy mode for the agent)
    OBVIOUS_VIOLATIONS = [
        ("HEALTHCARE", "Secret D1abet3s Cure", "Reverse it today with this root extract!", "A bottle of pills with no label.", True),
        ("FINANCIAL", "Garanteed 500% Returns", "Invest in LlamaCoin now. Zero risk.", "A stack of cash and a rocket emoji.", True)
    ]

    # 2. Multimodal Traps (Text is safe, but the image violates policy)
    # The agent MUST use `analyze_image` to catch these!
    MULTIMODAL_TRAPS = [
        ("HEALTHCARE", "Achieve Your Dream Body", "Hard work and our new program will get you there.", "Before and after weight loss comparison photos. (Meta Policy Violation)", True),
        ("FINANCIAL", "Get Approved in Minutes", "We help you navigate the complex world of finance.", "Fake government ID and a 'Guaranteed Loan Approval' stamp.", True),
        ("HEALTHCARE", "Stay Focused All Day", "Our study supplement helps you crush your exams.", "Close up of an Adderall prescription pill.", True)
    ]

    # 3. Subtle "Dog Whistle" Violations (Hard mode)
    SUBTLE_VIOLATIONS = [
        ("HEALTHCARE", "Optimize Metabolic Markers", "Our proprietary GLP-1 analog ships directly to your door without a visit.", "A sleek, branded injection pen.", True), # Selling prescription drugs
        ("FINANCIAL", "Beat the System", "Leverage our algorithmic trading bot to exploit market inefficiencies.", "A complex trading chart with green arrows pointing up.", True) # High-pressure/deceptive claims
    ]

    # 4. Perfectly Safe Ads (To prevent the agent from just rejecting everything)
    SAFE_ADS = [
        ("NONE", "Spring Collection 2026", "Check out our latest seasonal apparel.", "A model wearing a floral dress.", False),
        ("NONE", "Learn Python Online", "Master coding with our 10-week interactive bootcamp.", "A laptop screen showing clean Python code.", False),
        ("NONE", "Local Bakery Opening", "Come get fresh sourdough bread this Saturday!", "A warm, freshly baked loaf of bread.", False),
        ("FINANCIAL", "Plan For Retirement", "Speak with our certified fiduciaries to secure your future.", "A happy older couple walking on a beach.", False), # Safe financial ad
        ("HEALTHCARE", "Daily Multivitamin", "Support your immune system with Vitamin C and Zinc.", "A bottle of standard, FDA-approved daily vitamins.", False) # Safe health ad
    ]

    @staticmethod
    def generate_random_ad(task_id: str = None, difficulty: str = "random"):
        # Default to completely random if no task is specified
        if difficulty == "easy" and not task_id:
            scenario_type = random.choices(["obvious", "safe"], weights=[0.4, 0.6], k=1)[0]
        elif difficulty == "hard" and not task_id:
             scenario_type = random.choices(["multimodal", "subtle", "targeting_trap"], weights=[0.4, 0.4, 0.2], k=1)[0]
        else:
            scenario_type = random.choices(
                ["obvious", "multimodal", "subtle", "safe", "targeting_trap"],
                weights=[0.15, 0.25, 0.20, 0.30, 0.10],
                k=1
            )[0]

        # Initialize defaults
        category, headline, body, vlm_desc, is_violation = "NONE", "", "", "", False

        # 🧠 TASK OVERRIDES: Force specific scenarios based on the Meta Grader's request
        if task_id == "task_1_healthcare":
            pool = [ad for ad in AdGenerator.OBVIOUS_VIOLATIONS + AdGenerator.SUBTLE_VIOLATIONS if ad[0] == "HEALTHCARE"]
            category, headline, body, vlm_desc, is_violation = random.choice(pool)
            scenario_type = "forced"
            
        elif task_id == "task_2_financial":
            pool = [ad for ad in AdGenerator.OBVIOUS_VIOLATIONS + AdGenerator.SUBTLE_VIOLATIONS if ad[0] == "FINANCIAL"]
            category, headline, body, vlm_desc, is_violation = random.choice(pool)
            scenario_type = "forced"
            
        elif task_id == "task_3_multimodal":
            scenario_type = "multimodal"
            
        elif task_id == "task_4_targeting":
            scenario_type = "targeting_trap"

        # --- Base Logic Execution ---
        age_limit = random.randint(18, 55)

        if scenario_type == "obvious":
            category, headline, body, vlm_desc, is_violation = random.choice(AdGenerator.OBVIOUS_VIOLATIONS)
        elif scenario_type == "multimodal":
            category, headline, body, vlm_desc, is_violation = random.choice(AdGenerator.MULTIMODAL_TRAPS)
        elif scenario_type == "subtle":
            category, headline, body, vlm_desc, is_violation = random.choice(AdGenerator.SUBTLE_VIOLATIONS)
        elif scenario_type == "safe":
            category, headline, body, vlm_desc, is_violation = random.choice(AdGenerator.SAFE_ADS)
        elif scenario_type == "targeting_trap":
            category = "FINANCIAL"
            headline = "Start Your First Investment Portfolio"
            body = "Learn the basics of stock trading with our beginner-friendly app."
            vlm_desc = "A smartphone showing a simple stock graph."
            age_limit = 15 # The trap!
            is_violation = True

        # Apply policy drift / adversarial escalation
        if difficulty == "hard" or random.random() < 0.3:
            def apply_drift(text):
                replacements = {'a': '@', 'e': '3', 'i': '1', 'o': '0', 's': '$'}
                for k, v in replacements.items():
                    if random.random() < 0.4:
                        text = text.replace(k, v).replace(k.upper(), v)
                return text
            headline = apply_drift(headline)
            body = apply_drift(body)

        # Trust score logic: correlated with violation but not purely random
        if is_violation:
            trust_score = round(random.uniform(0.1, 0.45), 2)
        else:
            trust_score = round(random.uniform(0.55, 0.99), 2)
            
        # ✅ FIX: Indented to be inside the generate_random_ad function!
        ad_id = str(uuid.uuid4())[:8]
        # Remove ground truth leak
        advertiser_id = ad_id
        
        # Add landing_risk_keywords based on category and violation
        if is_violation:
            if category == "HEALTHCARE":
                kw_pool = ["cure", "guarantee", "miracle", "secret", "diet"]
            elif category == "FINANCIAL":
                kw_pool = ["risk-free", "guaranteed", "crypto", "100x", "return"]
            else:
                kw_pool = ["scam", "urgent", "wire", "click"]
        else:
            kw_pool = ["info", "contact", "about", "terms", "privacy", "team"]
            
        landing_risk_keywords = random.sample(kw_pool, k=min(2, len(kw_pool)))

        return {
            "ad_id": ad_id,
            "advertiser_id": advertiser_id,
            "task_id": task_id,                                       
            "headline": headline,
            "body_text": body,
            "advertiser_trust_score": trust_score,
            "targeting_data": {"age_min": age_limit, "geo": "US"},   
            "domain_age_days": random.randint(30, 1800),               
            "image_url": f"https://mock-meta.com/img/{uuid.uuid4()}.jpg",
            "ground_truth": is_violation,
            "category": category,
            "vlm_desc": vlm_desc,
            "landing_risk_keywords": landing_risk_keywords
        }