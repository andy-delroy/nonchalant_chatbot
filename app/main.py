#v1.5 (fix 15)
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from app.thismydb.database import SessionLocal, engine
from app.schemas.models import Asset, Location, Department, Status, User
from app.schemas.models import Asset
from transformers import pipeline
from pydantic import BaseModel
from app.utils.utils import extract_asset_tag
from .ml_model import nlp_model

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

# Fixed get_db function that correctly yields a session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.on_event("startup")
async def startup_event():
    # Initialize the NLP pipeline during startup
    app.state.qa_pipeline = pipeline("question-answering", model=nlp_model.get_model(), tokenizer=nlp_model.get_tokenizer())


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
    else:
        # General information query
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
    """Analyze the user's question using both rule-based and AI approaches"""
    question_lower = question.lower()
    print(f"Analyzing intent for: {question_lower}")

    # Check for direct asset tag lookup
    import re
    uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
    found_tags = re.findall(uuid_pattern, question_lower)
    
    if found_tags:
        return "asset_lookup_by_tag", {"asset_tag": found_tags[0]}
    
    # Prepare context for AI model, including the query itself
    locations = [loc.name for loc in db.query(Location).all()]
    departments = [dept.name for dept in db.query(Department).all()]
    statuses = [status.name for status in db.query(Status).all()]
    context_locations = f"Query: {question_lower}. Possible locations: {', '.join(loc.lower() for loc in locations)}."
    context_departments = f"Query: {question_lower}. Possible departments: {', '.join(dept.lower() for dept in departments)}."
    context_statuses = f"Query: {question_lower}. Possible statuses: {', '.join(status.lower() for status in statuses)}."
    
    # Define possible intents and filters
    filters = {}
    count_indicators = ["how many", "count", "total number", "number of"]
    if any(indicator in question_lower for indicator in count_indicators):
        intent = "asset_count"
    else:
        intent = "asset_filter"

    # Extract filters using AI, one entity type at a time
    # Step 1: Extract location
    nlp_input_location = {
        "question": f"What location is mentioned in this query: {question}",
        "context": context_locations
    }
    result_location = app.state.qa_pipeline(question=nlp_input_location["question"], context=nlp_input_location["context"])
    ai_answer_location = result_location['answer'].lower()
    ai_score_location = result_location['score']
    print(f"AI Location Answer: {ai_answer_location}, Score: {ai_score_location}")

    for loc in locations:
        loc_lower = loc.lower()
        if loc_lower in question_lower and loc_lower in ai_answer_location:
            location = db.query(Location).filter(Location.name.ilike(loc)).first()
            if location:
                filters["location_id"] = location.id
                print(f"AI detected location: {loc} (ID: {location.id})")
                break
    if "location_id" not in filters:
        for loc in locations:
            loc_lower = loc.lower()
            if loc_lower in ai_answer_location and ai_score_location > 0.5:  # Raise threshold for fallback
                location = db.query(Location).filter(Location.name.ilike(loc)).first()
                if location:
                    filters["location_id"] = location.id
                    print(f"AI detected location: {loc} (ID: {location.id})")
                    break

    # Step 2: Extract department
    nlp_input_department = {
        "question": f"What department is mentioned in this query: {question}",
        "context": context_departments
    }
    result_department = app.state.qa_pipeline(question=nlp_input_department["question"], context=nlp_input_department["context"])
    ai_answer_department = result_department['answer'].lower()
    ai_score_department = result_department['score']
    print(f"AI Department Answer: {ai_answer_department}, Score: {ai_score_department}")

    for dept in departments:
        dept_lower = dept.lower()
        # Only apply if the department is in the query OR the AI is very confident
        if (dept_lower in question_lower and dept_lower in ai_answer_department) or (dept_lower in ai_answer_department and ai_score_department > 0.9):
            department = db.query(Department).filter(Department.name.ilike(dept)).first()
            if department:
                filters["department_id"] = department.id
                print(f"AI detected department: {dept} (ID: {department.id})")
                break

    # Step 3: Extract status
    nlp_input_status = {
        "question": f"What status is mentioned in this query: {question}",
        "context": context_statuses
    }
    result_status = app.state.qa_pipeline(question=nlp_input_status["question"], context=nlp_input_status["context"])
    ai_answer_status = result_status['answer'].lower()
    ai_score_status = result_status['score']
    print(f"AI Status Answer: {ai_answer_status}, Score: {ai_score_status}")

    for status in statuses:
        status_lower = status.lower()
        # Only apply if the status is in the query OR the AI is very confident
        if (status_lower in question_lower and status_lower in ai_answer_status) or (status_lower in ai_answer_status and ai_score_status > 0.9):
            status_obj = db.query(Status).filter(Status.name.ilike(status)).first()
            if status_obj:
                filters["status_id"] = status_obj.id
                print(f"AI detected status: {status} (ID: {status_obj.id})")
                break

    # Rule-based fallback for conditions
    condition_values = {
        "new": "New", 
        "good": "Good",
        "fair": "Fair",
        "poor": "Poor",
        "excellent": "Excellent"
    }
    
    for value, db_value in condition_values.items():
        if value in question_lower:
            filters["condition"] = db_value
            print(f"Rule-based detected condition: {db_value}")
    
    # Rule-based status ID detection
    status_match = re.search(r'status\s+id\s+(\d+)', question_lower)
    if status_match:
        filters["status_id"] = int(status_match.group(1))
        print(f"Rule-based detected status ID: {filters['status_id']}")

    # If no filters are found, fall back to general query
    if not filters:
        return "general_query", {}
    
    print(f"Detected intent: {intent}, Extracted filters: {filters}")
    return intent, filters


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
    
    # Apply filters
    if "condition" in filters:
        if filters["condition"] is True:  # Just the keyword without value
            print("Condition keyword detected without specific value")
        else:
            print(f"Filtering by condition: {filters['condition']}")
            # Use exact match first
            query = query.filter(Asset.condition == filters["condition"])
    
    if "status_id" in filters:
        print(f"Filtering by status_id: {filters['status_id']}")
        query = query.filter(Asset.status_id == filters["status_id"])
    
    # if "location" in filters and filters["location"] is not True:
    #     print(f"Filtering by location: {filters['location']}")
    #     query = query.filter(Asset.location_id == filters["location"])

    if "location_id" in filters:
        if filters["location_id"] is None:
            return QueryResponse(answer="Location not found in the database.")
        print(f"Filtering by location_id: {filters['location_id']}")
        query = query.filter(Asset.location_id == filters["location_id"])
    
    # if "department" in filters and filters["department"] is not True:
    #     print(f"Filtering by department: {filters['department']}")
    #     query = query.filter(Asset.department_id == filters["department"])
    
    if "department_id" in filters:
        if filters["department_id"] is None:
            return QueryResponse(answer="Department not found in the database.")
        print(f"Filtering by department_id: {filters['department_id']}")
        query = query.filter(Asset.department_id == filters["department_id"])
        
    # Execute query and format results
    assets = query.all()  # Limit to 5 results
    print(f"Query returned {len(assets)} assets")
    
    if not assets:
        return QueryResponse(answer="No assets found matching your criteria.")
    
    result = "Found the following assets:\n"
    # for i, asset in enumerate(assets, 1):
    #     result += f"{i}. Asset: {asset.name}, Tag: {asset.asset_tag}"
    #     if hasattr(asset, 'condition') and asset.condition:
    #         result += f", Condition: {asset.condition}"
    #     if hasattr(asset, 'status_id') and asset.status_id:
    #         result += f", Status ID: {asset.status_id}"
    #     result += "\n"
    for i, asset in enumerate(assets, 1):
        result += f"{i}. Asset: {asset.name}, Tag: {asset.asset_tag}"
        if hasattr(asset, 'condition') and asset.condition:
            result += f", Condition: {asset.condition}"
        if hasattr(asset, 'status_id') and asset.status_id:
            result += f", Status ID: {asset.status_id}"
        if hasattr(asset, 'location') and asset.location:
            result += f", Location: {asset.location.name}"
        if hasattr(asset, 'department') and asset.department:
            result += f", Department: {asset.department.name}"
        if hasattr(asset, 'status') and asset.status:
            result += f", Status: {asset.status.name}"
        result += "\n"
    
    if len(assets) == 5:
        result += "\nShowing the first 5 results. There may be more matching assets."
    
    return QueryResponse(answer=result)

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

def handle_general_query(question, db):
    """Handle general questions about assets"""
    # For this, we can use the QA pipeline as before
    context = " ".join([f"Asset: {asset.name}, Tag: {asset.asset_tag}, Serial: {asset.serial}, Condition: {asset.condition}" 
                      for asset in db.query(Asset).limit(50).all()])
    
    nlp_input = {"question": question, "context": context}
    result = app.state.qa_pipeline(nlp_input)
    
    return QueryResponse(answer=result['answer'])

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