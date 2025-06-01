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
from app.services.ner_service import detect_uuid  # already defined
from app.services.llm_service import normalize_intent
from app.services.ner_service import merge_filters

#refactoring goes brrrrr
from app.services.asset_service import (
    handle_asset_filter_query,
    handle_asset_count_query,
    handle_asset_checkout,
    handle_asset_tag_lookup,
    handle_general_query,
)

from app.services.llm_service import call_llm_agent
from app.services.ner_service import analyze_query_intent

from app.models.pydantics import QueryResponse, QueryRequest, AssetItem
from app.core.context import get_context, save_context

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
    model_path = os.path.join(base_dir, "fine_tuned_lower_case")
    print(f"Loading model path: {model_path}")
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

    # UUID override: if found, shortcut to asset lookup
    uuid = detect_uuid(question)
    if uuid:
        return handle_asset_tag_lookup(uuid, db)
    
    # ---------- 1. figure out intent & filters ----------
    if mode == "llm":
        # Before LLM call
        reset_keywords = ["reset", "clear filter", "show everything", "start over", "all assets"]
        back_keywords = ["give me back all", "return all assets", "give back all"]

        # Context logic for LLM
        if any(kw in question.lower() for kw in reset_keywords):
            context["filters"] = {}
            print("🧹 LLM: Resetting all filters before call.")

        elif any(kw in question.lower() for kw in back_keywords):
            prev = context.get("filters", {})
            if "location_id" in prev:
                context["filters"] = {"location_id": prev["location_id"]}
                print("⏪ LLM: Keeping only location filter before call.")
            else:
                context["filters"] = {}
                print("⏪ LLM: No location found, full reset.")
        
        result   = await call_llm_agent(question, db, context)
        intent = result.get("intent", "general_query")
        intent = normalize_intent(result.get("intent", "general_query"))
        # filters = result.get("filters", {})
        filters = merge_filters(context.get("filters", {}), result.get("filters", {}))
        print(f"⭐ LLM Intent: {intent}, Filters: {filters}\n")
    else:
        #not llm then NER mode loh
        intent, filters = analyze_query_intent(question, db, context, app.state.ner_pipeline)
        print(f"⭐ NER Intent: {intent}, Filters: {filters}\n")

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
    print(f"🎯 Final intent: {intent}, filters: {filters}")
    print("----------------------------------------------------------\n")
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

#I can probably delete this funciton let's keep it for now
# def suggest_similar_location(detected_name: str, db_names: list, threshold=70, limit=3):
#     matches = process.extract(detected_name, db_names, limit=limit)
#     suggestions = [name for name, score in matches if score >= threshold]
#     return suggestions

#These are from inside the analyze_query_intent function
# It tries to reconstruct any sequence of tokens that have "LOCATION" in their label — regardless of B- or I-.
# But you've already done this way better using group_entities() and match_entity_to_filter(...).
    # location_spans = []
    # current_location = []
    # for entity in entities:
    #     if entity["entity"].endswith("LOCATION"):
    #         current_location.append(entity["word"])
    #     else:
    #         if current_location:
    #             location_spans.append(" ".join(current_location).replace(" ##", ""))
    #             current_location = []
    # if current_location:
    #     location_spans.append(" ".join(current_location).replace(" ##", ""))

#     This is your fuzzy fallback + suggestion logic, triggered only if location_id is still missing.
#     If your match_entity_to_filter() already:
#     Uses fuzzywuzzy.extractOne
#     Matches "branch office" to your DB locations
#     Fails silently if it doesn't meet the threshold
    # for span in location_spans:
    #     if "location_id" not in filters:
    #         match, score = process.extractOne(span.lower(), [loc.lower() for loc in locations])
    #         if score >= 80:
    #             loc = next(l for l in locations if l.lower() == match)
    #             location = db.query(Location).filter(Location.name.ilike(loc)).first()
    #             filters["location_id"] = location.id
    #             print(f"NER fuzzy matched location: {loc} (ID: {location.id})")
    #         else:
    #             suggestions = suggest_similar_location(span, locations)
    #             if suggestions:
    #                 print(f"❓ Unresolved location '{span}', did you mean: {suggestions}")
    #                 return "unresolved_location", {
    #                     "suggestions": suggestions,
    #                     "unmatched_location": span
    #                 }
    # Suggest fallback if no location_id and NER gave us location-like entities