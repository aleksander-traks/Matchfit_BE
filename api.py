import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor, as_completed

# import helpers from your existing file
from matchfit_app import (
    _call_with_fallback,
    generate_overview_via_llm,
    build_client_overview_from_sel,
    call_llm_match,
)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


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

    # ---- concurrent scoring to avoid super-long requests ----
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
            # Don't kill the whole request if one expert blows up
            return {
                "expert_id": ex["id"],
                "match_score": 0,
                "reason1": f"Error scoring expert: {e}",
                "reason2": "",
            }

    results = []
    max_workers = min(8, max(1, len(experts)))  # safe cap

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(score_one, ex) for ex in experts]
            for fut in as_completed(futures):
                results.append(fut.result())
    except Exception as e:
        # If something really bad happens, return a clean 500 JSON
        return jsonify({"error": f"Failed to score experts: {e}"}), 500

    # sort by score desc just in case
    results.sort(key=lambda r: r["match_score"], reverse=True)

    return jsonify({"matches": results})


@app.get("/")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))