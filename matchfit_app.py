import os
import json
import time
import random
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# ---------------- env ----------------
load_dotenv()

APP_TITLE = "MatchFit: Single-Table Tagging (Overview → Expert Match)"

CLIENTS_PATH = os.getenv("CLIENTS_PATH", "./Matchfit - Client.csv")
EXPERTS_PATH  = os.getenv("EXPERTS_PATH", "./experts-10.csv")

# columns in client CSV
CLIENT_FIELDS = [
    "Training Experience",
    "Goals",
    "How many times per week can you workout",
    "Chronic Disease",
    "Injuries",
    "Weight",
]

SINGLE_SELECT = {"Training Experience", "How many times per week can you workout", "Weight"}
MULTI_SELECT = {"Goals", "Chronic Disease", "Injuries"}

# ---------------- matching prompts ----------------

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

# ---------------- overview-generation prompts (LLM) ----------------

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
If no injuries/chronic issues are present, state that plainly and bias toward habits/progression.

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
- If none exist, say so and emphasize steady progression/habits.
- Echo the opener style from the examples (“Wants to …”).
- No headings, no emojis, no quotes, no markdown.
"""

# ---------------- helpers ----------------

def tx(v: Any) -> Optional[str]:
    if pd.isna(v): return None
    s = str(v).strip()
    return s if s and s.lower() not in ("nan","none") else None

def split_multi(s: Optional[str]) -> List[str]:
    if not s: return []
    out = []
    for p in str(s).replace("|",",").replace(";",",").split(","):
        p = p.strip()
        if p: out.append(p)
    return out

def ensure_required(df: pd.DataFrame) -> None:
    missing = [c for c in CLIENT_FIELDS if c not in df.columns]
    if missing:
        raise ValueError(f"Client CSV missing required columns: {missing}")

def unique_tags(series: pd.Series) -> List[str]:
    seen = {}
    for v in series.dropna():
        for t in split_multi(v):
            if t not in seen:
                seen[t] = True
    return list(seen.keys())

def build_expert_overview_from_row(row: pd.Series) -> str:
    # if expert_overview already present, prefer it
    pre = tx(row.get("expert_overview"))
    if pre: return pre
    fields_order = [
        "name","role","title","specialty","focus",
        "certifications","certs","methods","modalities",
        "experience_years","years_experience",
        "clinical_experience","conditions","populations",
        "tools","approach","about"
    ]
    pieces = []
    for k in fields_order:
        val = tx(row.get(k))
        if val:
            pieces.append(f"{k.replace('_',' ').title()}: {val}")
    if not pieces:
        for k, val in row.items():
            vv = tx(val)
            if vv: pieces.append(f"{k}: {vv}")
    return " | ".join(pieces) if pieces else "General fitness/physio expert."

def _get_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set. Put it in .env or use the sidebar override.")
    return key

# --- model helpers ---
VALID_MODELS = [
    "gpt-5-mini",       # small 5-series
    "gpt-4o-mini",      # fast/cheap, good JSON
    "gpt-4o",           # stronger multimodal
    "gpt-4.1-mini",
]
ALIASES = {
    "mini": "gpt-4o-mini",
    "4o-mini": "gpt-4o-mini",
    "4o": "gpt-4o",
    "gpt-5.1-mini": "gpt-5-mini",  # alias typo
}
def resolve_model(user_input: str) -> str:
    if not user_input: return VALID_MODELS[0]
    m = user_input.strip()
    if m in VALID_MODELS: return m
    return ALIASES.get(m, VALID_MODELS[0])

# ---------------- OpenAI calls ----------------

def _extract_text_from_responses(resp) -> str:
    # 0) Flat text provided by SDK
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    # 1) Walk the rich output structure
    try:
        pieces = []
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", None) == "message":
                for c in getattr(item, "content", []) or []:
                    ttype = getattr(c, "type", None)
                    if ttype in ("output_text", "text"):
                        t = getattr(c, "text", None)
                        if hasattr(t, "value"):  # some SDKs put the string in .text.value
                            t = t.value
                        if isinstance(t, str) and t.strip():
                            pieces.append(t)
        if pieces:
            return "".join(pieces).strip()
    except Exception:
        pass

    # 2) Legacy compat (chat.completions style)
    try:
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""

def _responses_create(client: OpenAI, model: str, messages: list):
    """
    Responses API wrapper.
    - gpt-5* models: no temperature param (deterministic decoding).
    - 4-series: include temperature=0.
    """
    kwargs = {"model": model, "input": messages}
    if not model.startswith("gpt-5"):
        kwargs["temperature"] = 0
    return client.responses.create(**kwargs)

def _sleep_with_jitter(base: float, attempt: int):
    time.sleep((base ** attempt) + random.uniform(0, 0.25))

FALLBACK_CHAIN = ["gpt-5-mini", "gpt-4o-mini", "gpt-4o"]

def _call_with_fallback(call_fn, primary_model: str, *args, **kwargs):
    tried = []
    for m in [primary_model] + [x for x in FALLBACK_CHAIN if x != primary_model]:
        try:
            return call_fn(model=m, *args, **kwargs)
        except Exception as e:
            msg = str(e).lower()
            tried.append((m, msg))
            # hopeless → try next model immediately
            if any(k in msg for k in ["model_not_found", "does not exist", "unsupported parameter", "not enabled"]):
                continue
            # transient → jitter and retry same model once
            if any(k in msg for k in ["rate limit", "timeout", "overloaded", "server error", "429", "502", "503"]):
                _sleep_with_jitter(1.5, len(tried))
                continue
            # other error: stop
            raise
    raise RuntimeError(f"All models failed: {tried[-1][0]} → {tried[-1][1]}")

def call_llm_match(client_overview: str, expert_overview: str, model: str) -> Dict[str, Any]:
    client = OpenAI(api_key=_get_api_key())
    user_content = USER_PROMPT_TEMPLATE.format(CLIENT_TEXT=client_overview, EXPERT_TEXT=expert_overview)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    # API call
    resp = _responses_create(client, model, messages)
    text = _extract_text_from_responses(resp)
    if not text:
        raise RuntimeError("Empty response from model")

    # robust JSON parse
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
    return {"match": int(data.get("match", 0)), "reasons": reasons}

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
    client = OpenAI(api_key=_get_api_key())
    facts = _facts_from_sel(sel)

    # few-shot messages for Responses API
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

# --------- single unified table (side-by-side lists) ---------

PAIR_COLS = [
    ("Training Experience", "TE ✓"),
    ("Goals", "Goals ✓"),
    ("How many times per week can you workout", "Per-week ✓"),
    ("Chronic Disease", "Chronic ✓"),
    ("Injuries", "Injuries ✓"),
    ("Weight", "Weight ✓"),
]

def build_side_by_side_grid(clients_df: pd.DataFrame) -> pd.DataFrame:
    # collect options per field
    pools: Dict[str, List[str]] = {name: unique_tags(clients_df[name]) for name, _ in PAIR_COLS}
    # find max rows needed
    max_rows = max((len(v) for v in pools.values()), default=0)
    # pad each list to max_rows with ""
    padded = {k: (v + [""]*(max_rows-len(v))) for k,v in pools.items()}
    # construct dataframe with label + checkbox column per field
    data: Dict[str, List[Any]] = {}
    for name, check_col in PAIR_COLS:
        data[name] = padded[name]
        data[check_col] = [False]*max_rows
    return pd.DataFrame(data)

def enforce_single_select(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for name, check_col in PAIR_COLS:
        if name in SINGLE_SELECT:
            true_idxs = out.index[out[check_col] == True].tolist()
            if len(true_idxs) > 1:
                keep = true_idxs[0]
                for i in true_idxs[1:]:
                    out.at[i, check_col] = False
    return out

def selections_from_side_by_side(df: pd.DataFrame) -> Dict[str, List[str]]:
    sel: Dict[str, List[str]] = {}
    for name, check_col in PAIR_COLS:
        vals = []
        for i, checked in enumerate(df[check_col].tolist()):
            if checked:
                label = str(df.at[i, name]).strip()
                if label:
                    vals.append(label)
        if name in SINGLE_SELECT and len(vals) > 1:
            vals = vals[:1]
        sel[name] = vals
    return sel

# --------- legacy simple builder (fallback) ---------

def build_client_overview_from_sel(sel: Dict[str, List[str]]) -> str:
    parts = []
    te = (sel.get("Training Experience") or [None])[0]
    if te: parts.append(f"Training experience: {te}.")
    goals = sel.get("Goals") or []
    if goals: parts.append(f"Goals: {', '.join(goals)}.")
    perw = (sel.get("How many times per week can you workout") or [None])[0]
    if perw: parts.append(f"Weekly availability: {perw} sessions.")
    chronic = sel.get("Chronic Disease") or []
    if chronic: parts.append(f"Chronic disease: {', '.join(chronic)}.")
    injuries = sel.get("Injuries") or []
    if injuries: parts.append(f"Injuries: {', '.join(injuries)}.")
    weight = (sel.get("Weight") or [None])[0]
    if weight: parts.append(f"Weight context: {weight}.")
    return " ".join(parts).strip() or "General fitness client; no further details provided."

# ---------------- caching ----------------

@st.cache_data
def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# ---------------- app ----------------

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("One table only: click the checkboxes next to each label. Singles enforce one; multis allow many.")

    # load
    if not os.path.exists(CLIENTS_PATH):
        st.error(f"Missing {CLIENTS_PATH}.")
        st.stop()
    if not os.path.exists(EXPERTS_PATH):
        st.error(f"Missing {EXPERTS_PATH}.")
        st.stop()

    try:
        clients_df = _load_csv(CLIENTS_PATH)
        experts_df = _load_csv(EXPERTS_PATH)
    except Exception as e:
        st.error(f"Failed to read CSVs: {e}")
        st.stop()

    try:
        ensure_required(clients_df)
    except Exception as e:
        st.error(str(e)); st.stop()

    with st.sidebar:
        model_raw = st.text_input("OpenAI model", value="gpt-5-mini")
        model = resolve_model(model_raw)
        api_override = st.text_input("Override OPENAI_API_KEY (optional)", type="password")
        if api_override:
            os.environ["OPENAI_API_KEY"] = api_override
        top_n = st.number_input("Show top N experts", min_value=1, max_value=1000, value=10, step=1)
        shorten = st.toggle("Shorten outputs (≤600 chars)", value=False)

    # init grid once
    if "grid" not in st.session_state:
        st.session_state["grid"] = build_side_by_side_grid(clients_df)

    st.subheader("1) Click-to-tag table")

    # form batches edits so we don't rerun on every tick
    with st.form("tagging_form", clear_on_submit=False):
        edited = st.data_editor(
            st.session_state["grid"],
            use_container_width=True,
            column_config={
                "Training Experience": st.column_config.TextColumn("Training Experience", disabled=True),
                "Goals": st.column_config.TextColumn("Goals", disabled=True),
                "How many times per week can you workout": st.column_config.TextColumn("How many times per week can you workout", disabled=True),
                "Chronic Disease": st.column_config.TextColumn("Chronic Disease", disabled=True),
                "Injuries": st.column_config.TextColumn("Injuries", disabled=True),
                "Weight": st.column_config.TextColumn("Weight", disabled=True),
                "TE ✓": st.column_config.CheckboxColumn("TE ✓", help="Single-select"),
                "Goals ✓": st.column_config.CheckboxColumn("Goals ✓", help="Multi-select"),
                "Per-week ✓": st.column_config.CheckboxColumn("Per-week ✓", help="Single-select"),
                "Chronic ✓": st.column_config.CheckboxColumn("Chronic ✓", help="Multi-select"),
                "Injuries ✓": st.column_config.CheckboxColumn("Injuries ✓", help="Multi-select"),
                "Weight ✓": st.column_config.CheckboxColumn("Weight ✓", help="Single-select"),
            },
            num_rows="fixed",
            key="one_table_editor",
        )
        submitted = st.form_submit_button("Apply selections")

    if submitted:
        st.session_state["grid"] = enforce_single_select(edited)

    st.subheader("2) Create overview")
    if "client_overview" not in st.session_state:
        st.session_state["client_overview"] = ""

    if st.button("Create overview"):
        selections = selections_from_side_by_side(st.session_state["grid"])
        if shorten:
            global OVERVIEW_USER_TMPL
            original = OVERVIEW_USER_TMPL
            OVERVIEW_USER_TMPL = OVERVIEW_USER_TMPL.strip() + "\n\nHard cap: 600 characters."
        try:
            overview = _call_with_fallback(
                lambda **k: generate_overview_via_llm(selections, **k),
                model
            )
        except Exception as e:
            overview = build_client_overview_from_sel(selections)
            st.warning(f"LLM overview failed ({e}); used simple fallback.")
        finally:
            if shorten:
                OVERVIEW_USER_TMPL = original
        st.session_state["client_overview"] = overview
        st.success("Client overview created (editable below).")

    client_overview = st.text_area("Client overview (editable)", value=st.session_state["client_overview"], height=180)

    st.subheader("3) Match with experts")
    if st.button("Match with experts"):
        if not client_overview.strip():
            st.error("Create the client overview first.")
            st.stop()

        # prefer existing expert_overview column; otherwise synthesize
        exp_overviews = []
        for _, r in experts_df.iterrows():
            pre = tx(r.get("expert_overview"))
            exp_overviews.append(pre if pre else build_expert_overview_from_row(r))

        ex_local = experts_df.copy()
        ex_local["__expert_overview__"] = exp_overviews

        # ---------- CONCURRENT SCORING ----------
        def _score_one(i, r):
            eov = r["__expert_overview__"]
            try:
                data = _call_with_fallback(
                    lambda **k: call_llm_match(client_overview, eov, **k),
                    model
                )
            except Exception as e:
                data = {"match": 0, "reasons": [f"Error: {e}", ""]}
            row_payload = {
                "row_index": i,
                "match": data["match"],
                "reason_1": data["reasons"][0],
                "reason_2": data["reasons"][1],
            }
            for k in ["expert_id","name","certs","specialty","years_experience"]:
                if k in ex_local.columns:
                    row_payload[k] = r.get(k)
            return row_payload

        results = []
        max_workers = 5  # conservative default
        progress = st.progress(0.0, text="Scoring experts…")
        total = len(ex_local)
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_score_one, i, r) for i, r in ex_local.iterrows()]
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as e:
                    results.append({"row_index": -1, "match": 0, "reason_1": f"Error: {e}", "reason_2": ""})
                completed += 1
                progress.progress(completed/total, text=f"Scoring experts… {completed}/{total}")
        progress.empty()
        # ---------- END CONCURRENT SCORING ----------

        out = pd.DataFrame(results).sort_values("match", ascending=False).reset_index(drop=True)
        st.success("Matching complete.")
        st.dataframe(out.head(int(top_n)))
        st.download_button(
            "Download results CSV",
            out.to_csv(index=False).encode("utf-8"),
            file_name="match_results.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
