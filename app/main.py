#v1.7 (fix 15) best-trained model withe existing data
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from app.thismydb.database import SessionLocal, engine
from app.schemas.models import Asset, Location, Department, Status, User
from app.schemas.models import Asset
from transformers import pipeline
from pydantic import BaseModel
from app.utils.utils import extract_asset_tag
# from .ml_model import nlp_model
from fuzzywuzzy import process
import re
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# to display pretty in front end
from typing import List, Optional

# Create a new Pydantic model for asset results
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

# Adjust your QueryResponse to optionally include a list of assets
class QueryResponse(BaseModel):
    answer: str
    assets: Optional[List[AssetItem]] = None

#helper functions
def suggest_similar_location(detected_name: str, db_names: list, threshold=70, limit=3):
    """
    Suggest similar location names based on fuzzy string matching.
    """
    matches = process.extract(detected_name, db_names, limit=limit)
    suggestions = [name for name, score in matches if score >= threshold]
    return suggestions

# Fixed get_db function that correctly yields a session
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

@app.post("/query", response_model=QueryResponse)
async def query_assets(request: QueryRequest, db: Session = Depends(get_db)):
    question = request.question
    print(f"⭐💩 Processing question: {question}")
    
     # Check what assets are in the database
    # sample_assets = db.query(Asset).limit(3).all()
    # print("Sample assets in database:")
    # for asset in sample_assets:
    #     print(f"  - Name: {asset.name}, Condition: {asset.condition}, Status ID: {asset.status_id}")
    
    # Identify query intent and extract filters
    #PASSING IN DB HERE
    intent, filters = analyze_query_intent(question, db)
    print(f"⭐ Intent: {intent}, Filters: {filters}")
    
    # Handle different query intents
    if intent == "asset_lookup_by_tag":
        # Current logic for asset tag lookup
        return handle_asset_tag_lookup(filters.get("asset_tag"), db)
    elif intent == "asset_filter":
        # New logic for filtered queries (condition, location, etc)
        return handle_asset_filter_query(filters, db)
    elif intent == "asset_count":
        # Logic for counting assets
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
    
    # # Example joint query: Get assets with their location, department, and status
    # query = (db.query(Asset, Location, Department, Status)
    #          .join(Location, Asset.location_id == Location.id)
    #          .join(Department, Asset.department_id == Department.id)
    #          .join(Status, Asset.status_id == Status.id))

    # Use left outer joins for Department and Status to include assets with missing departments/statuses
    query = (db.query(Asset, Location, Department, Status)
             .join(Location, Asset.location_id == Location.id)
             .outerjoin(Department, Asset.department_id == Department.id)  # Left join
             .outerjoin(Status, Asset.status_id == Status.id))  # Left join
    
    # Apply filters based on the question
    if "branch office" in question.lower():
        query = query.filter(Location.name.ilike("Branch Office"))
    if "it" in question.lower():
        query = query.filter(Department.name.ilike("IT"))
    if "available" in question.lower():
        query = query.filter(Status.name.ilike("Available"))
    
    # results = query.limit(5).all()
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

# Add a debug endpoint
@app.post("/debug_query")
async def debug_query(request: QueryRequest, db: Session = Depends(get_db)):
    question = request.question
    
    # Get all assets for context
    assets = db.query(Asset).all()
    
    # PRINTING FOR SAMPLE OUTPUT DEBUGGING
    # print("Sample assets:")
    # for asset in assets[:3]:
    #     print(asset)
    
    # Create context
    context = " ".join([f"Asset: {asset.name}, Tag: {asset.asset_tag}, Serial: {asset.serial}" 
                       for asset in assets])
    
    # Log what we're sending to the model
    print(f"Question: {question}")
    print(f"Number of assets in context: {len(assets)}")
    print(f"Context sample: {context[:200]}...")  # Print first 200 chars
    
    # Process with QA pipeline
    nlp_input = {"question": question, "context": context}
    result = app.state.qa_pipeline(nlp_input)
    
    # Return detailed debug info
    return {
        "question": question,
        "num_assets": len(assets),
        "context_length": len(context),
        "answer": result['answer'],
        "score": result['score'],
        "start": result.get('start', None),
        "end": result.get('end', None)
    }

#Modify the function to use the DeepSeek model to extract entities (locations, departments, statuses) from the query, while keeping rule-based logic as a fallback

def analyze_query_intent(question, db: Session):
    """Analyze the user's question using NER and rule-based approaches"""
    question_lower = question.lower()
    print(f"Analyzing intent for: {question_lower}")

    # Check for direct asset tag lookup (rule-based)
    uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
    found_tags = re.findall(uuid_pattern, question_lower)
    if found_tags:
        return "asset_lookup_by_tag", {"asset_tag": found_tags[0]}

    # Define possible intents and filters
    filters = {}
    count_indicators = ["how many", "count", "total number", "number of"]
    if any(indicator in question_lower for indicator in count_indicators):
        intent = "asset_count"
    else:
        intent = "asset_filter"

    # Get possible values from the database
    locations = [loc.name for loc in db.query(Location).all()]
    departments = [dept.name for dept in db.query(Department).all()]
    statuses = [status.name for status in db.query(Status).all()]

    # Use NER with original case for better recognition
    entities = app.state.ner_pipeline(question)  # Pass original question, not lowercase
    print(f"NER Entities: {entities}")

    # Process NER output
    current_entity = []
    for entity in entities:
        if entity["entity"].startswith("B-"):
            if current_entity:  # Process previous entity
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
            current_entity = [entity["word"]]
        elif entity["entity"].startswith("I-"):
            current_entity.append(entity["word"])
    if current_entity:  # Process last entity
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

    condition_values = {"new": "New", "good": "Good", "fair": "Fair", "poor": "Poor", "excellent": "Excellent"}
    for value, db_value in condition_values.items():
        if value in question_lower:
            filters["condition"] = db_value
            print(f"Rule-based detected condition: {db_value}")

    status_match = re.search(r'status\s+id\s+(\d+)', question_lower)
    if status_match:
        filters["status_id"] = int(status_match.group(1))
        print(f"Rule-based detected status ID: {filters['status_id']}")
    
    # Look for unresolved NER LOCATION spans
    location_spans = []
    current_location = []

    for entity in entities:
        if entity["entity"].endswith("LOCATION"):
            current_location.append(entity["word"])
        else:
            if current_location:
                location_spans.append(" ".join(current_location).replace(" ##", ""))
                current_location = []

    # Append final if NER ends on a location
    if current_location:
        location_spans.append(" ".join(current_location).replace(" ##", ""))

    # Try to resolve each detected LOCATION span
    for span in location_spans:
        if "location_id" not in filters:
            match, score = process.extractOne(span.lower(), [loc.lower() for loc in locations])
            if score >= 80:
                # Still good — silently resolve
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

    # if not filters:
    #     return "general_query", {}

    print(f"Detected intent: {intent}, Extracted filters: {filters}")
    return intent, filters

# def analyze_query_intent(question, db: Session):
    """Analyze the user's question using NER and rule-based approaches"""
    question_lower = question.lower()
    print(f"Analyzing intent for: {question_lower}")

    # Check for direct asset tag lookup (rule-based)
    uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
    found_tags = re.findall(uuid_pattern, question_lower)
    if found_tags:
        return "asset_lookup_by_tag", {"asset_tag": found_tags[0]}

    filters = {}
    count_indicators = ["how many", "count", "total number", "number of"]
    intent = "asset_count" if any(i in question_lower for i in count_indicators) else "asset_filter"

    locations = [loc.name for loc in db.query(Location).all()]
    departments = [dept.name for dept in db.query(Department).all()]
    statuses = [status.name for status in db.query(Status).all()]

    entities = app.state.ner_pipeline(question)
    print(f"NER Entities: {entities}")

    # Process NER into filters
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

    # Rule-based fallbacks
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

    condition_values = {"new": "New", "good": "Good", "fair": "Fair", "poor": "Poor", "excellent": "Excellent"}
    for value, db_value in condition_values.items():
        if value in question_lower:
            filters["condition"] = db_value
            print(f"Rule-based detected condition: {db_value}")

    # Check for unresolved NER locations and suggest alternatives
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

    if intent == "asset_filter" and not filters and location_spans:
        for span in location_spans:
            match, score = process.extractOne(span.lower(), [loc.lower() for loc in locations])
            if score >= 80:
                loc = next(l for l in locations if l.lower() == match)
                location = db.query(Location).filter(Location.name.ilike(loc)).first()
                filters["location_id"] = location.id
                print(f"Fallback fuzzy matched location: {loc} (ID: {location.id})")
                return intent, filters
            else:
                suggestions = suggest_similar_location(span, locations)
                if suggestions:
                    print(f"❓ Unresolved location '{span}', did you mean: {suggestions}")
                    return "unresolved_location", {
                        "suggestions": suggestions,
                        "unmatched_location": span
                    }

    if not filters:
        return "general_query", {}

    print(f"Detected intent: {intent}, Extracted filters: {filters}")
    return intent, filters

def handle_general_query(question, db):
    """Handle general questions by querying the database directly"""
    query = db.query(Asset)
    question_lower = question.lower()

    # Simple rule-based extraction for general queries
    locations = [loc.name for loc in db.query(Location).all()]
    for loc in locations:
        if loc.lower() in question_lower:
            location = db.query(Location).filter(Location.name.ilike(loc)).first()
            query = query.filter(Asset.location_id == location.id)
            print(f"General query filter - Location: {loc} (ID: {location.id})")
            break

    assets = query.all()
    if not assets:
        return QueryResponse(answer="No assets found matching your query.")
    
    # Create text response
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
        result += "\n"
    
    # Build structured asset data
    asset_items = []
    for asset in assets:
        status_obj = db.query(Status).get(asset.status_id) if asset.status_id else None
        location_obj = db.query(Location).get(asset.location_id) if asset.location_id else None
        department_obj = db.query(Department).get(asset.department_id) if asset.department_id else None
        
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
    
    # Return both text and structured data
    return QueryResponse(answer=result.strip(), assets=asset_items)

def handle_asset_tag_lookup(asset_tag, db):
    """Handle direct asset tag lookup"""
    asset = db.query(Asset).filter(Asset.asset_tag == asset_tag).first()
    if not asset:
        return QueryResponse(answer=f"No asset found with tag {asset_tag}")
    
    return QueryResponse(answer=f"Asset Name: {asset.name}, Tag: {asset.asset_tag}, Serial: {asset.serial}, Purchase Date: {asset.purchase_date}")

# Modify this function to consistently return both text and structured asset data
def handle_asset_filter_query(filters, db):
    """Handle queries that filter assets by properties"""
    query = db.query(Asset)
    print(f"Applying filters: {filters}")
    
    # Apply filters
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
        
    # Execute query and format results
    assets = query.all()
    print(f"Query returned {len(assets)} assets")
    
    if not assets:
        return QueryResponse(answer="No assets found matching your criteria.")
    
    # Create a text summary
    result = f"Found {len(assets)} assets matching your criteria."
    
    # Build a list of AssetItem objects with complete information
    asset_items = []
    for asset in assets:
        status_obj = db.query(Status).get(asset.status_id) if asset.status_id else None
        location_obj = db.query(Location).get(asset.location_id) if asset.location_id else None
        department_obj = db.query(Department).get(asset.department_id) if asset.department_id else None
        
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
    # Always return both the text answer and structured asset data
    return QueryResponse(answer=result, assets=asset_items)

def handle_asset_count_query(filters, db):
    """Handle queries that count assets with certain properties"""
    query = db.query(Asset)
    
    # Apply the same filters as in handle_asset_filter_query
    if "condition" in filters and filters["condition"] is not True:
        query = query.filter(Asset.condition.ilike(f"%{filters['condition']}%"))

    if "location_id" in filters and filters["location_id"] is not None:
        query = query.filter(Asset.location_id == filters["location_id"])
    
    if "department_id" in filters and filters["department_id"] is not None:
        query = query.filter(Asset.department_id == filters["department_id"])
    
    if "status_id" in filters and filters["status_id"] is not None:
        query = query.filter(Asset.status_id == filters["status_id"])
    
    # Apply other filters...
    
    count = query.count()
    
    filter_desc = ", ".join([f"{k}: {v}" for k, v in filters.items() if v is not True])
    if filter_desc:
        return QueryResponse(answer=f"Found {count} assets matching criteria: {filter_desc}")
    else:
        return QueryResponse(answer=f"Total number of assets: {count}")

# def handle_general_query(question, db):
#     """Handle general questions about assets"""
#     # For this, we can use the QA pipeline as before
#     context = " ".join([f"Asset: {asset.name}, Tag: {asset.asset_tag}, Serial: {asset.serial}, Condition: {asset.condition}" 
#                       for asset in db.query(Asset).limit(50).all()])
    
#     nlp_input = {"question": question, "context": context}
#     result = app.state.qa_pipeline(nlp_input)
    
#     return QueryResponse(answer=result['answer'])

def extract_filters_from_question(question):
    """Extract various filters from a natural language question"""
    filters = {}
    
    # Condition detection
    conditions = ["new", "good", "fair", "poor", "excellent"]
    for condition in conditions:
        if condition in question.lower():
            filters["condition"] = condition.capitalize()
            break
    
    # Status ID detection
    import re
    status_match = re.search(r'status\s+id\s+(\d+)', question.lower())
    if status_match:
        filters["status_id"] = int(status_match.group(1))
    
    # Date detection - purchase date before/after
    date_after = re.search(r'(purchased|bought)\s+(after|since)\s+(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}|\d{4})', question.lower())
    if date_after:
        # Parse the date string and add to filters
        filters["purchase_date_after"] = date_after.group(3)
    
    # More filter extraction logic...
    
    return filters