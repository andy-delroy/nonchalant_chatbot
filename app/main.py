# LLM agent v1
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from app.thismydb.database import SessionLocal, engine
from app.schemas.models import Asset, Location, Department, Status, User
from transformers import pipeline
from pydantic import BaseModel
from app.utils.utils import extract_asset_tag
from fuzzywuzzy import process
import re
import os
import json
import ollama
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import redis.asyncio as redis

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for structured responses
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
    mode: str = "ner"  # defaults to ner if unspecified

class QueryResponse(BaseModel):
    answer: str
    assets: Optional[List[AssetItem]] = None

# Helper functions
def suggest_similar_location(detected_name: str, db_names: list, threshold=70, limit=3):
    matches = process.extract(detected_name, db_names, limit=limit)
    suggestions = [name for name, score in matches if score >= threshold]
    return suggestions

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.on_event("startup")
async def startup_event():
    # Load NER pipeline
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

    # No need to load LLM locally; we'll use Ollama's API
    print("Using Ollama's Mistral model via API")

    r = redis.from_url("redis://localhost:6379", decode_responses=True)
    print("this nigga should print...")
    print(await r.ping())   # should print True


async def call_llm_agent(question: str, db: Session):
    """
    Use Ollama's Mistral model to process the query and extract intent and filters.
    Returns a dictionary with intent and filters to map to functions.
    """
    # Get possible values from the database for context
    locations = [loc.name for loc in db.query(Location).all()]
    departments = [dept.name for dept in db.query(Department).all()]
    statuses = [status.name for status in db.query(Status).all()]
    managers = [user.username for user in db.query(User).all()]

    # Construct the prompt
    prompt = f"""
You are an AI agent for an asset management system. Your task is to analyze the user's query and extract the intent and filters to query a database. The system supports the following intents:
- "asset_lookup_by_tag": Lookup an asset by its tag (a UUID, e.g., 123e4567-e89b-12d3-a456-426614174000).
- "asset_filter": Filter assets by properties like location, status, department, condition, or manager.
- "asset_count": Count assets matching certain properties.
- "general_query": For queries that don't fit the above intents.

Available locations: {', '.join(locations)}
Available departments: {', '.join(departments)}
Available statuses: {', '.join(statuses)}
Available managers: {', '.join(managers)}
Conditions: New, Good, Fair, Poor, Excellent

For the query: "{question}"

Return a JSON object with the following structure:
- "intent": One of the supported intents.
- "filters": A dictionary of filters (e.g., {{"location": "Warehouse", "status": "Available"}}). Use the exact names from the available options above. If an entity doesn't match any available option, include it as "unmatched_entity" with its type (e.g., {{"unmatched_entity": "Quantum Hub", "type": "location"}}).

Example:
Query: "Give me all the assets from Warehouse that are Available"
Output: {{"intent": "asset_filter", "filters": {{"location": "Warehouse", "status": "Available"}}}}
"""

    # Call Ollama's Mistral API
    response = ollama.generate(
        model="mistral",
        prompt=prompt,
        options={"temperature": 0.1, "max_tokens": 150}
    )
    llm_output = response["response"]

    # Extract JSON from the response (remove preamble like "Query: ... Output: ")
    json_start = llm_output.find("{")
    if json_start != -1:
        json_str = llm_output[json_start:]
        try:
            result = json.loads(json_str)
        except json.JSONDecodeError:
            print(f"LLM response parsing failed: {llm_output}")
            return {"intent": "general_query", "filters": {}}
    else:
        print(f"LLM response parsing failed: {llm_output}")
        return {"intent": "general_query", "filters": {}}

    intent = result.get("intent", "general_query")
    filters = result.get("filters", {})

    # Map entity names to database IDs
    processed_filters = {}
    if "location" in filters:
        location_name = filters["location"]
        location = db.query(Location).filter(Location.name.ilike(location_name)).first()
        if location:
            processed_filters["location_id"] = location.id
        else:
            suggestions = suggest_similar_location(location_name, locations)
            return {"intent": "unresolved_location", "filters": {"suggestions": suggestions, "unmatched_location": location_name}}

    if "status" in filters:
        status_name = filters["status"]
        status = db.query(Status).filter(Status.name.ilike(status_name)).first()
        if status:
            processed_filters["status_id"] = status.id

    if "department" in filters:
        dept_name = filters["department"]
        department = db.query(Department).filter(Department.name.ilike(dept_name)).first()
        if department:
            processed_filters["department_id"] = department.id

    if "manager" in filters:
        manager_name = filters["manager"]
        user = db.query(User).filter(User.username.ilike(manager_name)).first()
        if user:
            processed_filters["user_id"] = user.id

    if "condition" in filters:
        condition = filters["condition"]
        if condition in ["New", "Good", "Fair", "Poor", "Excellent"]:
            processed_filters["condition"] = condition

    if "unmatched_entity" in filters:
        return {"intent": "unresolved_location", "filters": {
            "suggestions": suggest_similar_location(filters["unmatched_entity"], locations if filters.get("type") == "location" else []),
            "unmatched_location": filters["unmatched_entity"]
        }}

    return {"intent": intent, "filters": processed_filters}

@app.post("/query", response_model=QueryResponse)
async def query_assets(request: QueryRequest, db: Session = Depends(get_db)):
    question = request.question
    mode = request.mode.lower()
    print(f"⭐💩 Processing question: {question} [mode: {mode}]")
    
    # Check the preferred mode
    if mode == "llm":
        # Use the LLM agent with Ollama's Mistral
        result = await call_llm_agent(question, db)
        intent = result["intent"]
        filters = result["filters"]
        print(f"⭐ LLM Intent: {intent}, Filters: {filters}")
    else:
        # Use the NER pipeline (existing logic)
        intent, filters = analyze_query_intent(question, db)
        print(f"⭐ NER Intent: {intent}, Filters: {filters}")
    
    # Handle different query intents
    if intent == "asset_lookup_by_tag":
        return handle_asset_tag_lookup(filters.get("asset_tag"), db)
    elif intent == "asset_filter":
        return handle_asset_filter_query(filters, db)
    elif intent == "asset_count":
        return handle_asset_count_query(filters, db)
    elif intent == "unresolved_location":
        suggestions = filters.get("suggestions", [])
        unmatched = filters.get("unmatched_location", "that location")
        return QueryResponse(
            answer=f"I couldn't find '{unmatched}' as a valid location in the system. Did you mean: {', '.join(suggestions)}?",
            assets=[]
        )
    else:
        return handle_general_query(question, db)

# Test endpoint for joint queries
@app.post("/test_joint_query", response_model=QueryResponse)
async def test_joint_query(request: QueryRequest, db: Session = Depends(get_db)):
    question = request.question
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
async def debug_query(request: QueryRequest, db: Session = Depends(get_db)):
    question = request.question
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

def analyze_query_intent(question, db: Session):
    """Analyze the user's question using NER and rule-based approaches"""
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

    # Rule-based fallback
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

    print(f"Detected intent: {intent}, Extracted filters: {filters}")
    return intent, filters

def handle_general_query(question, db):
    """Handle general questions by querying the database directly"""
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
    """Handle direct asset tag lookup"""
    asset = db.query(Asset).filter(Asset.asset_tag == asset_tag).first()
    if not asset:
        return QueryResponse(answer=f"No asset found with tag {asset_tag}")
    
    return QueryResponse(answer=f"Asset Name: {asset.name}, Tag: {asset.asset_tag}, Serial: {asset.serial}, Purchase Date: {asset.purchase_date}")

def handle_asset_filter_query(filters, db):
    """Handle queries that filter assets by properties"""
    query = db.query(Asset)
    print(f"Applying filters: {filters}")
    
    if "condition" in filters and filters["condition"] is not True:
        print(f"Filtering by condition: {filters['condition']}")
        query = query.filter(Asset.condition == filters["condition"])
    
    if "status_id" in filters:
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
    """Handle queries that count assets with certain properties"""
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
        return QueryResponse(answer=f"Total number of assets: {count}")