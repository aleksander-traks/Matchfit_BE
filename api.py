# api.py

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor, as_completed

from matchfit_core import (
    _call_with_fallback,
    generate_overview_via_llm,
    build_client_overview_from_sel,
    call_llm_match,
)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# global thread pool → reused across requests
MAX_MATCH_WORKERS = int(os.getenv("MATCHFIT_MAX_WORKERS", "4"))
EXECUTOR = ThreadPoolExecutor(max_workers=MAX_MATCH_WORKERS)

# optional cap on experts per request (tunable via env, high by default)
MAX_EXPERTS_PER_REQUEST = int(os.getenv("MATCHFIT_MAX_EXPERTS", "1000"))

DEFAULT_MODEL = os.getenv("MATCHFIT_MODEL", "gpt-5-nano")


@app.post("/generate-overview")
def generate_overview():
    data = request.get_json(force=True) or {}

    sel = {
        "Training Experience": [data.get("training_experience")],
        "Goals": data.get("goals") or [],
        "How many times per week can you workout": (
            [str(data.get("sessions_per_week"))]
            if data.get("sessions_per_week") is not None
            else []
        ),
        "Chronic Disease": data.get("chronic_diseases") or [],
        "Injuries": data.get("injuries") or [],
        "Weight": [data.get("weight_goal")],
    }

    # If literally nothing selected, avoid LLM call
    if not any(sel.values()):
        overview = build_client_overview_from_sel(sel)
        return jsonify({"overview": overview})

    model = DEFAULT_MODEL

    try:
        overview = _call_with_fallback(
            lambda **k: generate_overview_via_llm(sel, **k),
            model,
        )
    except Exception:
        # fallback to simple builder
        overview = build_client_overview_from_sel(sel)

    return jsonify({"overview": overview})


@app.post("/match-experts")
def match_experts():
    data = request.get_json(force=True) or {}

    client_overview = data.get("client_overview", "")
    experts = data.get("experts", [])

    if not isinstance(client_overview, str) or not client_overview.strip():
        return jsonify({"error": "client_overview must be a non-empty string"}), 400
    if not isinstance(experts, list) or not experts:
        return jsonify({"error": "experts must be a non-empty list"}), 400

    # hard cap per request (does NOT change shape, just safety)
    experts = experts[:MAX_EXPERTS_PER_REQUEST]

    model = DEFAULT_MODEL

    def score_one(ex):
        try:
            m = _call_with_fallback(
                lambda **k: call_llm_match(client_overview, ex["overview"], **k),
                model,
            )
            return {
                "expert_id": ex["id"],
                "match_score": m["match"],
                "reason1": m["reasons"][0],
                "reason2": m["reasons"][1],
            }
        except Exception as e:
            # Don't kill whole request if one fails
            return {
                "expert_id": ex.get("id"),
                "match_score": 0,
                "reason1": f"Error scoring expert: {e}",
                "reason2": "",
            }

    results = []
    try:
        # all experts submitted to the global pool → concurrent scoring
        futures = [EXECUTOR.submit(score_one, ex) for ex in experts]
        for fut in as_completed(futures):
            results.append(fut.result())
    except Exception as e:
        return jsonify({"error": f"Failed to score experts: {e}"}), 500

    # sort by score desc
    results.sort(key=lambda r: r["match_score"], reverse=True)

    return jsonify({"matches": results})


@app.get("/")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
