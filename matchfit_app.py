# matchfit_core.py

import os
import json
import time
import random
from typing import List, Dict, Any, Optional

from openai import OpenAI

# ---------------- env + client ----------------

def _get_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return key

_client: Optional[OpenAI] = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=_get_api_key())
    return _client

# ---------------- prompts ----------------

SYSTEM_PROMPT = (
    "You are a deterministic matching engine for physio/fitness clients.\n"
    "Return ONLY JSON."
)

USER_PROMPT_TEMPLATE = '''
CLIENT_OVERVIEW:
"""{CLIENT_TEXT}"""

EXPERT_OVERVIEW:
"""{EXPERT_TEXT}"""

Task:
Give ONE holistic match score (0-100).
- If the client overview includes specific medical/musculoskeletal issues, emphasize clinical safety/effectiveness.
- Otherwise, emphasize goal achievement and coaching fit.
Always: note decisive overlaps (issues<->methods/certs, goals<->focus), and apply small penalties for obvious mismatches.

Output JSON ONLY (exact schema):
{{
  "match": <0-100>,
  "reasons": ["<short decisive reason #1>", "<short decisive reason #2>"]
}}
'''

OVERVIEW_SYSTEM = (
    "You write single-paragraph client overviews for physio/fitness intake. "
    "Be concise, rehab-aware, and deterministic. No lists, no headings, no emojis."
)

OVERVIEW_FEWSHOTS = [
    ("facts", {
        "training_experience": "1–3 years",
        "goals": ["weight loss", "general fitness"],
        "per_week": "3",
        "chronic": [], "injuries": [], "weight": "overweight"
    }),
    ("output",
     "Wants to reach a healthy weight and improve general fitness. Has no chronic diseases or injuries limiting activity, which means training can focus on steady progression and habit-building. With 1–3 years of experience, they’re familiar with basic technique and can safely work toward long-term endurance and conditioning."
    ),
    ("facts", {
        "training_experience": "3–5 years",
        "goals": ["less pain", "move easier"], "per_week": "4",
        "chronic": [], "injuries": []
    }),
    ("output",
     "Wants to experience less pain and move more comfortably. No listed chronic issues or injuries, allowing a broader approach emphasizing mobility, core stability, and stress reduction. Trains regularly for 3–5 years, showing good self-discipline and capacity for structured progressive overload."
    ),
    ("facts", {
        "training_experience": "1–3 years",
        "goals": ["get stronger"],
        "injuries": ["hamstring strain", "UCL sprain (skier’s thumb)"],
        "chronic": ["hypermobility spectrum disorder"]
    }),
    ("output",
     "Wants to get stronger. Deals with hypermobility spectrum disorder, which makes joints less stable and increases the risk of overextension during strength work. Also recovering from a pulled hamstring and a skier’s thumb (UCL sprain), both requiring caution in eccentric loading and grip-heavy movements. With 1–3 years of training, they can handle intermediate programs if supervision ensures correct movement control."
    ),
]

OVERVIEW_USER_TMPL = """
Create ONE short paragraph (no bullets, no lists) that matches the style of the examples. 
Keep clinical phrasing accurate, then add practical emphasis. Do not invent conditions. 
If no injuries or chronic issues are present, state that plainly and bias toward habits and progression.

FACTS
- Training experience: {te}
- Goals: {goals}
- Sessions per week: {perw}
- Chronic disease: {chronic}
- Injuries: {injuries}
- Weight: {weight}

Rules:
- 3–6 sentences total.
- If clinical flags exist, include them explicitly and note training implications.
- If none exist, say so and emphasize steady progression and habits.
- Echo the opener style from the examples (“Wants to …”).
- No headings, no emojis, no quotes, no markdown.
"""

# ---------------- responses helpers ----------------

def _extract_text_from_responses(resp) -> str:
    # 0) Flat text
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    # 1) Rich output
    try:
        pieces = []
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", None) == "message":
                for c in getattr(item, "content", []) or []:
                    ttype = getattr(c, "type", None)
                    if ttype in ("output_text", "text"):
                        t = getattr(c, "text", None)
                        if hasattr(t, "value"):
                            t = t.value
                        if isinstance(t, str) and t.strip():
                            pieces.append(t)
        if pieces:
            return "".join(pieces).strip()
    except Exception:
        pass

    # 2) Legacy-style
    try:
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""

def _responses_create(client: OpenAI, model: str, messages: list):
    kwargs = {"model": model, "input": messages}
    if not model.startswith("gpt-5"):
        kwargs["temperature"] = 0
    return client.responses.create(**kwargs)

def _sleep_with_jitter(base: float, attempt: int):
    time.sleep((base ** attempt) + random.uniform(0, 0.25))

FALLBACK_CHAIN = ["gpt-5-mini", "gpt-4o-mini", "gpt-4o"]

def _call_with_fallback(call_fn, primary_model: str, *args, **kwargs):
    """
    call_fn is something like: lambda **k: generate_overview_via_llm(sel, **k)
    It must accept model=... via kwargs.
    """
    tried = []
    chain = [primary_model] + [m for m in FALLBACK_CHAIN if m != primary_model]
    for m in chain:
        try:
            return call_fn(model=m, *args, **kwargs)
        except Exception as e:
            msg = str(e).lower()
            tried.append((m, msg))

            if any(k in msg for k in ["model_not_found", "does not exist", "unsupported parameter", "not enabled"]):
                continue

            if any(k in msg for k in ["rate limit", "timeout", "overloaded", "server error", "429", "502", "503"]):
                _sleep_with_jitter(1.5, len(tried))
                continue

            # fatal error → give up
            raise
    raise RuntimeError(f"All models failed: {tried[-1][0]} -> {tried[-1][1]}")

# ---------------- overview generation ----------------

def _facts_from_sel(sel: Dict[str, List[str]]) -> Dict[str, Any]:
    return {
        "te": (sel.get("Training Experience") or [None])[0],
        "goals": sel.get("Goals") or [],
        "perw": (sel.get("How many times per week can you workout") or [None])[0],
        "chronic": sel.get("Chronic Disease") or [],
        "injuries": sel.get("Injuries") or [],
        "weight": (sel.get("Weight") or [None])[0],
    }

def generate_overview_via_llm(sel: Dict[str, List[str]], model: str) -> str:
    client = get_client()
    facts = _facts_from_sel(sel)

    messages = [{"role": "system", "content": OVERVIEW_SYSTEM}]
    for role, payload in OVERVIEW_FEWSHOTS:
        if role == "facts":
            messages.append({"role": "user", "content": json.dumps(payload, ensure_ascii=False)})
        else:
            messages.append({"role": "assistant", "content": payload})

    user_msg = OVERVIEW_USER_TMPL.format(
        te=facts["te"] or "unspecified",
        goals=", ".join(facts["goals"]) or "unspecified",
        perw=facts["perw"] or "unspecified",
        chronic=", ".join(facts["chronic"]) or "none",
        injuries=", ".join(facts["injuries"]) or "none",
        weight=facts["weight"] or "unspecified",
    )
    messages.append({"role": "user", "content": user_msg})

    resp = _responses_create(client, model, messages)
    text = _extract_text_from_responses(resp)
    text = " ".join(text.split())
    if not text:
        raise RuntimeError("Empty response from model")
    return text

def build_client_overview_from_sel(sel: Dict[str, List[str]]) -> str:
    parts = []
    te = (sel.get("Training Experience") or [None])[0]
    if te:
        parts.append(f"Training experience: {te}.")
    goals = sel.get("Goals") or []
    if goals:
        parts.append(f"Goals: {', '.join(goals)}.")
    perw = (sel.get("How many times per week can you workout") or [None])[0]
    if perw:
        parts.append(f"Weekly availability: {perw} sessions.")
    chronic = sel.get("Chronic Disease") or []
    if chronic:
        parts.append(f"Chronic disease: {', '.join(chronic)}.")
    injuries = sel.get("Injuries") or []
    if injuries:
        parts.append(f"Injuries: {', '.join(injuries)}.")
    weight = (sel.get("Weight") or [None])[0]
    if weight:
        parts.append(f"Weight context: {weight}.")
    return " ".join(parts).strip() or "General fitness client; no further details provided."

# ---------------- expert matching ----------------

def call_llm_match(client_overview: str, expert_overview: str, model: str) -> Dict[str, Any]:
    client = get_client()
    user_content = USER_PROMPT_TEMPLATE.format(
        CLIENT_TEXT=client_overview,
        EXPERT_TEXT=expert_overview,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    resp = _responses_create(client, model, messages)
    text = _extract_text_from_responses(resp)
    if not text:
        raise RuntimeError("Empty response from model")

    try:
        data = json.loads(text)
    except Exception:
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e != -1 and e > s:
            data = json.loads(text[s:e+1])
        else:
            raise RuntimeError(f"Non-JSON model output: {text[:200]}...")

    reasons = (data.get("reasons") or [])[:2]
    reasons += [""] * (2 - len(reasons))

    return {
        "match": int(data.get("match", 0)),
        "reasons": reasons,
    }
