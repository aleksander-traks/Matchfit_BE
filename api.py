import os
from flask import Flask, request, jsonify

# import helpers from your existing file
from matchfit_app import (
    _call_with_fallback,
    generate_overview_via_llm,
    build_client_overview_from_sel,
    call_llm_match,
)

app = Flask(__name__)

@app.post("/generate-overview")
def generate_overview():
    data = request.get_json(force=True)

    sel = {
        "Training Experience": [data.get("training_experience")],
        "Goals": data.get("goals") or [],
        "How many times per week can you workout": [str(data.get("sessions_per_week"))],
        "Chronic Disease": data.get("chronic_diseases") or [],
        "Injuries": data.get("injuries") or [],
        "Weight": [data.get("weight_goal")],
    }

    model = "gpt-5-mini"

    try:
        overview = _call_with_fallback(
            lambda **k: generate_overview_via_llm(sel, **k),
            model,
        )
    except Exception:
        # simple fallback if LLM fails
        overview = build_client_overview_from_sel(sel)

    return jsonify({"overview": overview})


@app.post("/match-experts")
def match_experts():
    data = request.get_json(force=True)
    client_overview = data["client_overview"]
    experts = data["experts"]  # [{ "id": number, "overview": string }]

    model = "gpt-5-mini"
    results = []

    for ex in experts:
        m = _call_with_fallback(
            lambda **k: call_llm_match(client_overview, ex["overview"], **k),
            model,
        )
        results.append({
            "expert_id": ex["id"],
            "match_score": m["match"],
            "reason1": m["reasons"][0],
            "reason2": m["reasons"][1],
        })

    return jsonify({"matches": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))