import re
from sqlalchemy.orm import Session
from typing import Dict, Tuple, List
from fuzzywuzzy import process
from app.schemas.models import Asset, Location, Department, Status, User
from app.services.asset_service import suggest_similar_location

def detect_uuid(question: str) -> str | None:
    uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
    matches = re.findall(uuid_pattern, question.lower())
    return matches[0] if matches else None

# def merge_filters(previous: Dict, current: Dict) -> Dict:
#     result = previous.copy()
#     result.update({k: v for k, v in current.items() if v is not None})
#     return result

#utils
def merge_filters(prev: Dict, new: Dict) -> Dict:
    out = prev.copy()
    out.update({k: v for k, v in new.items() if v is not None})
    return out

def suggest_similar_location(name: str, all_names: List[str], threshold=70, limit=3):
    return [n for n, s in process.extract(name, all_names, limit=limit) if s >= threshold]


def suggest_fallback_entity(phrase, valid_choices, threshold=80):
    """
    Tries to fuzzy match a phrase against valid choices.
    Returns None if match is good.
    Returns suggestions list if no good match is found.
    """
    match = process.extractOne(phrase.lower(), [c.lower() for c in valid_choices])
    if match and match[1] >= threshold:
        return None  # good match, no suggestion needed

    # Get top 3 suggestions above 60
    raw = process.extract(phrase.lower(), [c.lower() for c in valid_choices], limit=5)
    suggestions = [c for c, score in raw if score >= 60]
    return suggestions if suggestions else None

#NEW HELPER FUNCTION
def match_entity_to_filter(label: str, phrase: str, db, filters: dict, lookup_map: dict):
    """
    Matches a NER-labeled phrase to the corresponding database entry using fuzzywuzzy.
    Adds the resolved ID to the filters dict under the appropriate field name.
    """
    print("From match_entity_to_filter method:")
    print(f"label: {label}, phrase: {phrase}, filters: {filters}")
    if label not in lookup_map:
        print(f"[SKIP] Label '{label}' not in lookup map.")
        return

    lookup = lookup_map[label]
    choices = lookup["choices"]
    model = lookup["model"]
    column = lookup["column"]
    field = lookup["field"]

    # Fuzzy match the extracted phrase against DB values
    match = process.extractOne(phrase.lower(), [c.lower() for c in choices])
    if match and match[1] >= 80:
        matched_string = match[0]
        actual = next(c for c in choices if c.lower() == matched_string)
        obj = db.query(model).filter(column.ilike(actual)).first()

        if obj:
            filters[field] = obj.id
            print(f"[NER MATCH] {label}: {actual} → {field} = {obj.id}\n")
    else:
        print(f"[NO MATCH] {label}: '{phrase}' did not meet threshold\n")

def group_entities(entities):
    grouped = []
    current = []
    print("From group_entities method:")

    for ent in entities:
        print(ent)
        if ent["entity"].startswith("B-"):
            if current:
                grouped.append(current)
            current = [ent]
        elif ent["entity"].startswith("I-") and current:
            current.append(ent)
        else:
            if current:
                grouped.append(current)
                current = []
    if current:
        grouped.append(current)
    
    print()
    return grouped


def apply_rule_based_fallbacks(question_lower, db, filters):

    print("From apply_rule_based_fallbacks_method")
    fallback_map = {
        "status_id": {
            "choices": db.query(Status).all(),
            "field": "status_id",
            "value_getter": lambda x: x.name,
            "model": Status,
            "column": Status.name
        },
        "department_id": {
            "choices": db.query(Department).all(),
            "field": "department_id",
            "value_getter": lambda x: x.name,
            "model": Department,
            "column": Department.name
        },
        "location_id": {
            "choices": db.query(Location).all(),
            "field": "location_id",
            "value_getter": lambda x: x.name,
            "model": Location,
            "column": Location.name
        },
        "user_id": {
            "choices": db.query(User).all(),
            "field": "user_id",
            "value_getter": lambda x: x.username,
            "model": User,
            "column": User.username
        },
    }

    for field, meta in fallback_map.items():
        if field not in filters:
            for obj in meta["choices"]:
                name = meta["value_getter"](obj)
                if name.lower() in question_lower:
                    filters[field] = obj.id
                    print(f"[RULE DETECT] {field}: {name} (ID: {obj.id})")
                    print(f"[RULE FALLBACK] {field} filled via keyword match → '{name}' (ID: {obj.id})")
                    break

    # Handle condition specifically
    condition_values = {
        "new": "New",
        "mint": "Mint",
        "good": "Good",
        "fair": "Fair",
        "poor": "Poor",
        "excellent": "Excellent"
    }

    for key, val in condition_values.items():
        if key in question_lower:
            filters["condition"] = val
            # print(f"[RULE DETECT] condition: {val}")

    # Explicit "status id 2" format
    match = re.search(r'status\\s+id\\s+(\\d+)', question_lower)
    if match:
        filters["status_id"] = int(match.group(1))
        # print(f"[RULE DETECT] status_id via regex: {filters['status_id']}\n ")


#the purpose of this function is to take a question and analyze it's intent and extract filters
def analyze_query_intent(question, db: Session, context: dict, ner_pipeline):
    question_lower = question.lower()
    print(f"Analyzing intent for: {question_lower}")

    #  If user slaps in an asset tag, skip everything — they clearly know what they want.
    uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
    found_tags = re.findall(uuid_pattern, question_lower)
    if found_tags:
        return "asset_lookup_by_tag", {"asset_tag": found_tags[0]}

    #filters to be populated. Scoped to this function.
    filters = {}
    count_indicators = ["how many", "count", "total number", "number of"]
    if any(indicator in question_lower for indicator in count_indicators):
        intent = "asset_count"
    else:
        intent = "asset_filter"

    checkout_indicators = ["check out", "checkout", "check-out"]
    if any(ci in question_lower for ci in checkout_indicators):
        intent = "asset_checkout"

    #pulled known entities from DB. Used later to match NER entity labels and fallback string search
    locations = [loc.name for loc in db.query(Location).all()]
    departments = [dept.name for dept in db.query(Department).all()]
    statuses = [status.name for status in db.query(Status).all()]
    managers = [user.username for user in db.query(User).all()]

    # runs the NER [{ "word": "HQ", "entity": "B-LOCATION" }]
    entities = ner_pipeline(question.lower())
    print("Raw NER output...")
    print(f"NER Entities: {entities}\n") 
    

    lookup_map = {
        "LOCATION": {
            "choices": locations,
            "model": Location,
            "column": Location.name,
            "field": "location_id"
        },
        "STATUS": {
            "choices": statuses,
            "model": Status,
            "column": Status.name,
            "field": "status_id"
        },
        "DEPARTMENT": {
            "choices": departments,
            "model": Department,
            "column": Department.name,
            "field": "department_id"
        },
        "MANAGER": {
            "choices": managers,
            "model": User,
            "column": User.username,
            "field": "user_id"
        }
    }

    grouped_entities = group_entities(entities)

    # for group in grouped_entities:
    #     phrase = " ".join(e["word"] for e in group).replace(" ##", "")
    #     label = group[0]["entity"].split("-")[1]  # Get "LOCATION", "STATUS", etc.
    #     match_entity_to_filter(label, phrase, db, filters, lookup_map)
    #we will transform the above block by adding confidence boundaries
    for group in grouped_entities:
        phrase = " ".join(e["word"] for e in group).replace(" ##", "")
        label = group[0]["entity"].split("-")[1]
        
        # Confidence threshold check
        avg_score = sum(e["score"] for e in group) / len(group)
        if avg_score < 0.4:
            print(f"[NER SKIP] '{phrase}' ({label}) skipped due to low avg score: {avg_score:.2f}")
            continue
        print()
        print("From Confidence Threadshold check")
        print(f"[NER MATCHING] '{phrase}' ({label}) with avg score: {avg_score:.2f}")
        match_entity_to_filter(label, phrase, db, filters, lookup_map)



    """
    this will not override anything. Rather, it will fill in the blanks as we are passing the filters dict that is returned from match_entity_to_filter
    """
    apply_rule_based_fallbacks(question_lower, db, filters)

    if "location_id" not in filters:
        for group in grouped_entities:
            label = group[0]["entity"].split("-")[1]
            if label == "LOCATION":
                phrase = " ".join(e["word"] for e in group).replace(" ##", "")
                suggestions = suggest_fallback_entity(phrase, locations)
                if suggestions:
                    print(f"❓ Unresolved location '{phrase}', did you mean: {suggestions}")
                    return "unresolved_location", {
                        "suggestions": suggestions,
                        "unmatched_location": phrase
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
    # question_lower = question.lower()
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
