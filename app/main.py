# LLM agent v1
from fastapi import FastAPI, Depends, Request, HTTPException
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy.orm import Session
from app.thismydb.database import SessionLocal, engine
from app.schemas.models import Asset, Location, Department, Status, User
from transformers import pipeline
from pydantic import BaseModel
from app.utils.utils import extract_asset_tag
from fuzzywuzzy import process
import re, textwrap
import os
import json
import ollama
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis
from typing import List, Optional, Dict

def _first_json_block(text: str) -> str | None:
    """Return the first {...}-balanced JSON chunk inside `text`, or None."""
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i        # mark the first opening brace
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : i + 1]        # inclusive slice
    return None


app = FastAPI()

# Session Middleware with secret key
app.add_middleware(SessionMiddleware, secret_key=os.environ.get("SESSION_KEY", "your-secret-key-here"))
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis setup with redis.asyncio
REDIS     = Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"),
                           decode_responses=True)
CTX_TTL   = 900  # seconds

async def get_context(request: Request) -> Dict:
    sid = request.session.setdefault("sid", os.urandom(8).hex())
    key = f"ctx:{sid}"
    data = await REDIS.get(key)
    ctx  = json.loads(data) if data else {}

    # 👀 DEBUG (remove later)
    print(f"[CTX] sid={sid}  loaded={ctx}")
    return ctx


async def save_context(request: Request, ctx: Dict):
    await REDIS.setex(f"ctx:{request.session['sid']}", CTX_TTL, json.dumps(ctx))

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.on_event("startup")
async def startup_event():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "fine_tuned_ner_v3")
    if not os.path.exists(model_path):
        raise RuntimeError(f"NER model directory not found at {model_path}")
    app.state.ner_pipeline = pipeline(
        "ner",
        model=model_path,
        tokenizer=model_path,
        device=0  # GPU
    )
    app.state.ner_pipeline.model.eval()
    print(f"NER pipeline loaded from {model_path}")
    print("Using Ollama's Mistral model via API")

#utils
def merge_filters(prev: Dict, new: Dict) -> Dict:
    out = prev.copy()
    out.update({k: v for k, v in new.items() if v is not None})
    return out

def suggest_similar_location(name: str, all_names: List[str], threshold=70, limit=3):
    return [n for n, s in process.extract(name, all_names, limit=limit) if s >= threshold]

# Pydantic models
class AssetItem(BaseModel):
    id: int
    name: str
    tag: str
    condition: Optional[str] = None
    status_id: Optional[int] = None
    status_name: Optional[str] = None
    location_name: Optional[str] = None
    department_name: Optional[str] = None

class QueryRequest(BaseModel):
    question: str
    mode: str = "ner"

class QueryResponse(BaseModel):
    answer: str
    assets: Optional[List[AssetItem]] = None
    checkout_url: Optional[str] = None

async def call_llm_agent(question: str, db: Session, context: Dict):
    print(f"[LLM]  previous_filters seen by the LLM = {context.get('filters')}")
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

Previous context filters: {context.get("filters", {})}

Query: "{question}"
"""
    llm_raw = ollama.generate("mistral", prompt, options={"temperature": 0.1, "max_tokens": 150})["response"]
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

    intent   = result.get("intent", "general_query")
    filters  = result.get("filters", {})
    pfilters = {}  # processed

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

    # ---- other filters ----
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

def analyze_query_intent(question, db: Session, context: dict):
    question_lower = question.lower()
    print(f"Analyzing intent for: {question_lower}")

    uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
    found_tags = re.findall(uuid_pattern, question_lower)
    if found_tags:
        return "asset_lookup_by_tag", {"asset_tag": found_tags[0]}

    filters = {}
    count_indicators = ["how many", "count", "total number", "number of"]
    if any(indicator in question_lower for indicator in count_indicators):
        intent = "asset_count"
    else:
        intent = "asset_filter"

    checkout_indicators = ["check out", "checkout", "check-out"]
    if any(ci in question_lower for ci in checkout_indicators):
        intent = "asset_checkout"

    locations = [loc.name for loc in db.query(Location).all()]
    departments = [dept.name for dept in db.query(Department).all()]
    statuses = [status.name for status in db.query(Status).all()]
    managers = [user.username for user in db.query(User).all()]

    entities = app.state.ner_pipeline(question)
    print(f"NER Entities: {entities}")

    current_entity = []
    for entity in entities:
        if entity["entity"].startswith("B-"):
            if current_entity:
                entity_text = " ".join(current_entity).replace(" ##", "")
                label = entities[len(current_entity) - 1]["entity"].split("-")[1]
                if label == "LOCATION":
                    location_matches = process.extractOne(entity_text.lower(), [loc.lower() for loc in locations])
                    if location_matches and location_matches[1] > 80:
                        loc = next(l for l in locations if l.lower() == location_matches[0])
                        location = db.query(Location).filter(Location.name.ilike(loc)).first()
                        filters["location_id"] = location.id
                        print(f"NER detected location: {loc} (ID: {location.id})")
                elif label == "STATUS":
                    status_matches = process.extractOne(entity_text.lower(), [s.lower() for s in statuses])
                    if status_matches and status_matches[1] > 80:
                        status = next(s for s in statuses if s.lower() == status_matches[0])
                        status_obj = db.query(Status).filter(Status.name.ilike(status)).first()
                        filters["status_id"] = status_obj.id
                        print(f"NER detected status: {status} (ID: {status_obj.id})")
                elif label == "DEPARTMENT":
                    dept_matches = process.extractOne(entity_text.lower(), [d.lower() for d in departments])
                    if dept_matches and dept_matches[1] > 80:
                        dept = next(d for d in departments if d.lower() == dept_matches[0])
                        department = db.query(Department).filter(Department.name.ilike(dept)).first()
                        filters["department_id"] = department.id
                        print(f"NER detected department: {dept} (ID: {department.id})")
                elif label == "MANAGER":
                    manager_matches = process.extractOne(entity_text.lower(), [m.lower() for m in managers])
                    if manager_matches and manager_matches[1] > 80:
                        manager = next(m for m in managers if m.lower() == manager_matches[0])
                        user = db.query(User).filter(User.username.ilike(manager)).first()
                        filters["user_id"] = user.id
                        print(f"NER detected manager: {manager} (ID: {user.id})")
            current_entity = [entity["word"]]
        elif entity["entity"].startswith("I-"):
            current_entity.append(entity["word"])
    if current_entity:
        entity_text = " ".join(current_entity).replace(" ##", "")
        label = entities[-1]["entity"].split("-")[1]
        if label == "LOCATION":
            location_matches = process.extractOne(entity_text.lower(), [loc.lower() for loc in locations])
            if location_matches and location_matches[1] > 80:
                loc = next(l for l in locations if l.lower() == location_matches[0])
                location = db.query(Location).filter(Location.name.ilike(loc)).first()
                filters["location_id"] = location.id
                print(f"NER detected location: {loc} (ID: {location.id})")
        elif label == "STATUS":
            status_matches = process.extractOne(entity_text.lower(), [s.lower() for s in statuses])
            if status_matches and status_matches[1] > 80:
                status = next(s for s in statuses if s.lower() == status_matches[0])
                status_obj = db.query(Status).filter(Status.name.ilike(status)).first()
                filters["status_id"] = status_obj.id
                print(f"NER detected status: {status} (ID: {status_obj.id})")
        elif label == "DEPARTMENT":
            dept_matches = process.extractOne(entity_text.lower(), [d.lower() for d in departments])
            if dept_matches and dept_matches[1] > 80:
                dept = next(d for d in departments if d.lower() == dept_matches[0])
                department = db.query(Department).filter(Department.name.ilike(dept)).first()
                filters["department_id"] = department.id
                print(f"NER detected department: {dept} (ID: {department.id})")
        elif label == "MANAGER":
            manager_matches = process.extractOne(entity_text.lower(), [m.lower() for m in managers])
            if manager_matches and manager_matches[1] > 80:
                manager = next(m for m in managers if m.lower() == manager_matches[0])
                user = db.query(User).filter(User.username.ilike(manager)).first()
                filters["user_id"] = user.id
                print(f"NER detected manager: {manager} (ID: {user.id})")

    for status in statuses:
        if status.lower() in question_lower and "status_id" not in filters:
            status_obj = db.query(Status).filter(Status.name.ilike(status)).first()
            filters["status_id"] = status_obj.id
            print(f"Rule-based detected status: {status} (ID: {status_obj.id})")
    for dept in departments:
        if dept.lower() in question_lower and "department_id" not in filters:
            department = db.query(Department).filter(Department.name.ilike(dept)).first()
            filters["department_id"] = department.id
            print(f"Rule-based detected department: {dept} (ID: {department.id})")
    for loc in locations:
        if loc.lower() in question_lower and "location_id" not in filters:
            location = db.query(Location).filter(Location.name.ilike(loc)).first()
            filters["location_id"] = location.id
            print(f"Rule-based detected location: {loc} (ID: {location.id})")
    for manager in managers:
        if manager.lower() in question_lower and "user_id" not in filters:
            user = db.query(User).filter(User.username.ilike(manager)).first()
            filters["user_id"] = user.id
            print(f"Rule-based detected manager: {manager} (ID: {user.id})")

    condition_values = {"new": "New", "good": "Good", "fair": "Fair", "poor": "Poor", "excellent": "Excellent"}
    for value, db_value in condition_values.items():
        if value in question_lower:
            filters["condition"] = db_value
            print(f"Rule-based detected condition: {db_value}")

    status_match = re.search(r'status\s+id\s+(\d+)', question_lower)
    if status_match:
        filters["status_id"] = int(status_match.group(1))
        print(f"Rule-based detected status ID: {filters['status_id']}")

    location_spans = []
    current_location = []
    for entity in entities:
        if entity["entity"].endswith("LOCATION"):
            current_location.append(entity["word"])
        else:
            if current_location:
                location_spans.append(" ".join(current_location).replace(" ##", ""))
                current_location = []
    if current_location:
        location_spans.append(" ".join(current_location).replace(" ##", ""))

    for span in location_spans:
        if "location_id" not in filters:
            match, score = process.extractOne(span.lower(), [loc.lower() for loc in locations])
            if score >= 80:
                loc = next(l for l in locations if l.lower() == match)
                location = db.query(Location).filter(Location.name.ilike(loc)).first()
                filters["location_id"] = location.id
                print(f"NER fuzzy matched location: {loc} (ID: {location.id})")
            else:
                suggestions = suggest_similar_location(span, locations)
                if suggestions:
                    print(f"❓ Unresolved location '{span}', did you mean: {suggestions}")
                    return "unresolved_location", {
                        "suggestions": suggestions,
                        "unmatched_location": span
                    }

    if intent == "asset_filter":
        # if user did NOT mention status/asset this turn, wipe them
        if "status_id" not in filters and "available" not in question_lower:
            filters["status_id"] = None          # causes merge to overwrite
        if "asset_id" not in filters:
            filters["asset_id"] = None
    # Merge with previous context
    # filters = merge_filters(context.get("filters", {}), filters)
    #smarter merging
    # With this glorious hybrid AI/rule blend:
    question_lower = question.lower()
    reset_keywords = ["all assets", "reset", "show everything", "start over", "clear filter"]
    back_keywords = ["give me back all", "back all the assets", "return all the assets", "give back all"]

    # CASE 1: Hard reset if explicitly asked
    if any(kw in question_lower for kw in reset_keywords):
        print("🧹 Resetting all filters by user command.")
        filters = {}

    # CASE 2: User says "give me back all" → drop recent filters, keep location
    elif any(kw in question_lower for kw in back_keywords):
        prev_filters = context.get("filters", {})
        if "location_id" in prev_filters:
            filters = {"location_id": prev_filters["location_id"]}
            print("⏪ Keeping only location filter from previous context.")
        else:
            filters = {}
            print("⏪ No prior location found — full reset triggered.")

    # CASE 3: Normal merge
    else:
        filters = merge_filters(context.get("filters", {}), filters)

    if intent == "asset_filter" and "location_id" not in filters and "location_id" in context.get("filters", {}):
        if "only the available ones" in question.lower() or "available ones" in question.lower():
            filters["location_id"] = context["filters"]["location_id"]
            print(f"Reused location_id {filters['location_id']} from previous context")

    print(f"Detected intent: {intent}, Extracted filters: {filters}")
    return intent, filters

@app.post("/query", response_model=QueryResponse)
async def query_assets(
    request: Request,
    db: Session = Depends(get_db),
    context: dict = Depends(get_context)
):
    body = await request.json()
    question = body.get("question")
    mode = body.get("mode", "ner").lower()
    print(f"⭐💩 Processing question: {question} [mode: {mode}]")

    # ---------- 1. figure out intent & filters ----------
    if mode == "llm":
        result   = await call_llm_agent(question, db, context)
        intent   = result["intent"]
        filters  = result["filters"]
        print(f"⭐ LLM Intent: {intent}, Filters: {filters}")
    else:
        intent, filters = analyze_query_intent(question, db, context)
        print(f"⭐ NER Intent: {intent}, Filters: {filters}")

    # ---------- 2. run the proper handler & get a resp ----------
    if intent == "asset_lookup_by_tag":
        resp = handle_asset_tag_lookup(filters.get("asset_tag"), db)
    elif intent == "asset_filter":
        resp = handle_asset_filter_query(filters, db)
    elif intent == "asset_count":
        resp = handle_asset_count_query(filters, db)
    elif intent == "asset_checkout":
        resp = handle_asset_checkout(filters, context, db) # new
    elif intent == "unresolved_location":
        suggestions = filters.get("suggestions", [])
        unmatched   = filters.get("unmatched_location", "that location")
        resp = QueryResponse(
            answer=f"I couldn't find '{unmatched}' as a valid location in the system. "
                   f"Did you mean: {', '.join(suggestions)}?",
            assets=[]
        )
    else:
        resp = handle_general_query(question, db)

    # ---------- 3.  ▓ stash single-asset context  ▓ ----------
    if intent in ["asset_filter", "general_query"] and len(getattr(resp, "assets", [])) == 1:
        asset_id = resp.assets[0].id
        context["last_asset_id"] = asset_id      # remember for follow-ups
        filters["asset_id"]      = asset_id      # so checkout handler can reuse it

    # ---------- 4. persist the (possibly updated) context ----------
    await save_context(request, {
        "filters"        : filters,
        "last_asset_id"  : context.get("last_asset_id")
    })

    # ---------- 5. return ----------
    return resp


# Test endpoint for joint queries
@app.post("/test_joint_query", response_model=QueryResponse)
async def test_joint_query(request: Request, db: Session = Depends(get_db), context: dict = Depends(get_context)):
    body = await request.json()  # Await the JSON body
    question = body.get("question")
    print(f"Testing joint query: {question}")
    
    query = (db.query(Asset, Location, Department, Status)
             .join(Location, Asset.location_id == Location.id)
             .outerjoin(Department, Asset.department_id == Department.id)
             .outerjoin(Status, Asset.status_id == Status.id))
    
    if "warehouse" in question.lower():
        query = query.filter(Location.name.ilike("Warehouse"))
    if "it" in question.lower():
        query = query.filter(Department.name.ilike("IT"))
    if "available" in question.lower():
        query = query.filter(Status.name.ilike("Available"))
    
    results = query.all()
    print(f"Query returned {len(results)} assets.🗣️🗣️🗣️🗣️")
    if not results:
        return QueryResponse(answer="No assets found matching the test criteria.")
    
    result = "Found the following assets with joint table data:\n"
    for i, (asset, location, department, status) in enumerate(results, 1):
        result += (f"{i}. Asset: {asset.name}, Tag: {asset.asset_tag}, "
                   f"Location: {location.name}, "
                   f"Department: {department.name if department else 'None'}, "
                   f"Status: {status.name if status else 'None'}\n")
    
    return QueryResponse(answer=result)

# Debug endpoint
@app.post("/debug_query")
async def debug_query(request: Request, db: Session = Depends(get_db), context: dict = Depends(get_context)):
    body = await request.json()  # Await the JSON body
    question = body.get("question")
    assets = db.query(Asset).all()
    context = " ".join([f"Asset: {asset.name}, Tag: {asset.asset_tag}, Serial: {asset.serial}" 
                       for asset in assets])
    
    print(f"Question: {question}")
    print(f"Number of assets in context: {len(assets)}")
    print(f"Context sample: {context[:200]}...")
    
    return {
        "question": question,
        "num_assets": len(assets),
        "context_length": len(context),
        "answer": "Debug endpoint not fully implemented (missing qa_pipeline).",
        "score": 0.0,
        "start": None,
        "end": None
    }

def suggest_similar_location(detected_name: str, db_names: list, threshold=70, limit=3):
    matches = process.extract(detected_name, db_names, limit=limit)
    suggestions = [name for name, score in matches if score >= threshold]
    return suggestions

def handle_general_query(question, db):
    query = db.query(Asset)
    question_lower = question.lower()

    locations = [loc.name for loc in db.query(Location).all()]
    for loc in locations:
        if loc.lower() in question_lower:
            location = db.query(Location).filter(Location.name.ilike(loc)).first()
            query = query.filter(Asset.location_id == location.id)
            print(f"General query filter - Location: {loc} (ID: {location.id})")
            break

    statuses = [status.name for status in db.query(Status).all()]
    for status in statuses:
        if status.lower() in question_lower:
            status_obj = db.query(Status).filter(Status.name.ilike(status)).first()
            query = query.filter(Asset.status_id == status_obj.id)
            print(f"General query filter - Status: {status} (ID: {status_obj.id})")
            break

    assets = query.all()
    if not assets:
        return QueryResponse(answer="No assets found matching your query.")
    
    result = "Found the following assets general:\n"
    for i, asset in enumerate(assets, 1):
        result += f"{i}. Asset: {asset.name}, Tag: {asset.asset_tag}"
        if asset.condition:
            result += f", Condition: {asset.condition}"
        if asset.status_id:
            status = db.query(Status).filter(Status.id == asset.status_id).first()
            result += f", Status: {status.name if status else 'Unknown'}"
        if asset.location_id:
            location = db.query(Location).filter(Location.id == asset.location_id).first()
            result += f", Location: {location.name if location else 'Unknown'}"
        if asset.department_id:
            department = db.query(Department).filter(Department.id == asset.department_id).first()
            result += f", Department: {department.name if department else 'None'}"
        if asset.user_id:
            user = db.query(User).filter(User.id == asset.user_id).first()
            result += f", Managed by: {user.username if user else 'None'}"
        result += "\n"
    
    asset_items = []
    for asset in assets:
        status_obj = db.query(Status).get(asset.status_id) if asset.status_id else None
        location_obj = db.query(Location).get(asset.location_id) if asset.location_id else None
        department_obj = db.query(Department).get(asset.department_id) if asset.department_id else None
        user_obj = db.query(User).get(asset.user_id) if asset.user_id else None
        
        asset_items.append(AssetItem(
            id=asset.id,
            name=asset.name,
            tag=asset.asset_tag,
            condition=asset.condition,
            status_id=asset.status_id,
            status_name=(status_obj.name if status_obj else None),
            location_name=(location_obj.name if location_obj else None),
            department_name=(department_obj.name if department_obj else None)
        ))
    
    print("called General")
    return QueryResponse(answer=result.strip(), assets=asset_items)

def handle_asset_tag_lookup(asset_tag, db):
    asset = db.query(Asset).filter(Asset.asset_tag == asset_tag).first()
    if not asset:
        return QueryResponse(answer=f"No asset found with tag {asset_tag}")
    
    return QueryResponse(answer=f"Asset Name: {asset.name}, Tag: {asset.asset_tag}, Serial: {asset.serial}, Purchase Date: {asset.purchase_date}")

def handle_asset_filter_query(filters, db):
    query = db.query(Asset)
    print(f"Applying filters: {filters}")
    
    if "condition" in filters and filters["condition"] is not True:
        print(f"Filtering by condition: {filters['condition']}")
        query = query.filter(Asset.condition == filters["condition"])
    
    if filters.get("status_id") is not None:
        print(f"Filtering by status_id: {filters['status_id']}")
        query = query.filter(Asset.status_id == filters["status_id"])
    
    if "location_id" in filters:
        if filters["location_id"] is None:
            return QueryResponse(answer="Location not found in the database.")
        print(f"Filtering by location_id: {filters['location_id']}")
        query = query.filter(Asset.location_id == filters["location_id"])
    
    if "department_id" in filters:
        if filters["department_id"] is None:
            return QueryResponse(answer="Department not found in the database.")
        print(f"Filtering by department_id: {filters['department_id']}")
        query = query.filter(Asset.department_id == filters["department_id"])
    
    if "user_id" in filters:
        print(f"Filtering by user_id: {filters['user_id']}")
        query = query.filter(Asset.user_id == filters["user_id"])
        
    assets = query.all()
    print(f"Query returned {len(assets)} assets")
    
    if not assets:
        return QueryResponse(answer="No assets found matching your criteria.")
    
    result = f"Found {len(assets)} assets matching your criteria."
    
    asset_items = []
    for asset in assets:
        status_obj = db.query(Status).get(asset.status_id) if asset.status_id else None
        location_obj = db.query(Location).get(asset.location_id) if asset.location_id else None
        department_obj = db.query(Department).get(asset.department_id) if asset.department_id else None
        user_obj = db.query(User).get(asset.user_id) if asset.user_id else None
        
        asset_items.append(AssetItem(
            id=asset.id,
            name=asset.name,
            tag=asset.asset_tag,
            condition=asset.condition,
            status_id=asset.status_id,
            status_name=(status_obj.name if status_obj else None),
            location_name=(location_obj.name if location_obj else None),
            department_name=(department_obj.name if department_obj else None)
        ))
    
    print("called filter")
    return QueryResponse(answer=result, assets=asset_items)

def handle_asset_count_query(filters, db):
    query = db.query(Asset)
    
    if "condition" in filters and filters["condition"] is not True:
        query = query.filter(Asset.condition.ilike(f"%{filters['condition']}%"))

    if "location_id" in filters and filters["location_id"] is not None:
        query = query.filter(Asset.location_id == filters["location_id"])
    
    if "department_id" in filters and filters["department_id"] is not None:
        query = query.filter(Asset.department_id == filters["department_id"])
    
    if "status_id" in filters and filters["status_id"] is not None:
        query = query.filter(Asset.status_id == filters["status_id"])
    
    if "user_id" in filters and filters["user_id"] is not None:
        query = query.filter(Asset.user_id == filters["user_id"])
    
    count = query.count()
    
    filter_desc = ", ".join([f"{k}: {v}" for k, v in filters.items() if v is not True])
    if filter_desc:
        return QueryResponse(answer=f"Found {count} assets matching criteria: {filter_desc}")
    else:
        return QueryResponse(answer=f"Total number of assets: {count}")# LLM agent v1


def handle_asset_checkout(filters: Dict,
                          context: Dict,
                          db: Session) -> QueryResponse:
    """
    Build the /assets/<id>/checkout URL for the requested asset.

    Priority order for figuring out which asset to check out:
    1.  filters["asset_id"]
    2.  filters["asset_tag"]
    3.  context["last_asset_id"]   ← value we stashed after a 1-asset result
    """
    asset = None

    # ① explicit id from filters
    if "asset_id" in filters:
        asset = db.query(Asset).get(filters["asset_id"])

    # ② explicit tag from filters
    elif "asset_tag" in filters:
        asset = db.query(Asset).filter(
            Asset.asset_tag == filters["asset_tag"]
        ).first()

    # ③ single-asset follow-up (what you needed)
    elif context.get("last_asset_id"):
        asset = db.query(Asset).get(context["last_asset_id"])

    # still nothing?  Tell the user.
    if not asset:
        return QueryResponse(
            answer="I couldn’t figure out which asset you want to check out."
        )

    url = f"/assets/{asset.id}/checkout"
    return QueryResponse(
        answer=f"Here you go – hit the checkout page for **{asset.name}**.",
        checkout_url=url
    )


