import re
import json
import ollama
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from app.schemas.models import Location, Department, Status, User
from app.services.asset_service import suggest_similar_location

def _first_json_block(text: str) -> str | None:
    """Return the first {...}-balanced JSON chunk inside `text`, or None."""
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : i + 1]
    return None

# Core LLM interface to process a user's question using Mistral (via Ollama)
async def call_llm_agent(question: str, db: Session, context: Dict):
    print(f"[LLM]  previous_filters seen by the LLM = {context.get('filters')}")

    # Get known values for fuzzy matching or validation in prompt
    locations   = [l.name for l in db.query(Location).all()]
    departments = [d.name for d in db.query(Department).all()]
    statuses    = [s.name for s in db.query(Status).all()]
    managers    = [u.username for u in db.query(User).all()]    

    prompt = f"""
You are an AI agent for an asset-tracking system. For the user query below, return a JSON like
{{"intent":"asset_filter","filters":{{"location":"Warehouse","status":"Available"}}}}.
If it's a follow-up and the user omits location, reuse location from previous context.

The system supports the following intents:
- "asset_lookup_by_tag"
- "asset_filter"
- "asset_count"
- "asset_checkout"  
- "general_query"

Use these if mentioned: locations = {locations}, statuses = {statuses}, departments = {departments}, managers = {managers}, - conditions = ["New", "Good", "Fair", "Poor", "Excellent", "Mint"]
Previous context filters: {context.get("filters", {})}

Query: "{question}"
"""
    # Call Mistral via Ollama and extract raw response  
    llm_raw = ollama.generate("mistral", prompt, options={"temperature": 0.1, "max_tokens": 150})["response"]

    #try to extract the JSON response
    json_chunk = _first_json_block(llm_raw)

    if json_chunk is None:
        print("LLM returned no brace-balanced JSON – defaulting to general_query")
        return {"intent": "general_query", "filters": {}}

    # remove // comments that Mistral sometimes adds
    json_chunk = re.sub(r"//.*", "", json_chunk)

    try:
        result = json.loads(json_chunk)
    except json.JSONDecodeError as e:
        print("LLM JSON parse failed:", e)
        return {"intent": "general_query", "filters": {}}

    #LLM result fields
    intent   = result.get("intent", "general_query")
    filters  = result.get("filters", {})
    pfilters = {}  # processed filters for our DB queries

    # ---- location id / name handling (bullet-proof) ----
    if "location_id" in filters and isinstance(filters["location_id"], int):
        pfilters["location_id"] = filters["location_id"]

    elif "location" in filters:
        loc_val = filters["location"]
        if isinstance(loc_val, int):
            pfilters["location_id"] = loc_val
        elif isinstance(loc_val, str) and loc_val.isdigit():
            pfilters["location_id"] = int(loc_val)
        else:
            loc_obj = db.query(Location).filter(Location.name.ilike(loc_val)).first()
            if loc_obj:
                pfilters["location_id"] = loc_obj.id
            else:
                return {"intent": "unresolved_location",
                        "filters": {"suggestions": suggest_similar_location(loc_val, locations),
                                    "unmatched_location": loc_val}}

    # ---- other filters ---- (optional? these are not optional)
    if "status" in filters:
        stat = db.query(Status).filter(Status.name.ilike(filters["status"])).first()
        if stat:
            pfilters["status_id"] = stat.id
    if "department" in filters:
        dept = db.query(Department).filter(Department.name.ilike(filters["department"])).first()
        if dept:
            pfilters["department_id"] = dept.id
    if "manager" in filters:
        usr = db.query(User).filter(User.username.ilike(filters["manager"])).first()
        if usr:
            pfilters["user_id"] = usr.id
    if "condition" in filters:
        pfilters["condition"] = filters["condition"]

    # ---- follow-up location reuse ----
    if (intent == "asset_filter"
        and "location_id" not in pfilters
        and "location_id" in context.get("filters", {})
        and any(kw in question.lower() for kw in ["only the", "available ones", "ones"])):
        pfilters["location_id"] = context["filters"]["location_id"]

    return {"intent": intent, "filters": pfilters}

def normalize_intent(intent: str) -> str:
    """
    Normalize LLM intent output to match known backend handlers.
    """
    intent = intent.lower().strip()

    if intent.startswith("asset_filter"):
        return "asset_filter"
    elif intent.startswith("asset_count"):
        return "asset_count"
    elif intent.startswith("asset_lookup"):
        return "asset_lookup_by_tag"
    elif intent.startswith("asset_checkout"):
        return "asset_checkout"
    elif intent.startswith("unresolved_location"):
        return "unresolved_location"
    else:
        return "general_query"