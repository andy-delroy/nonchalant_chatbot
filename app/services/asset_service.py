from sqlalchemy.orm import Session
from app.schemas.models import Asset, Location, Department, Status, User
from app.models.pydantics import QueryResponse, QueryRequest, AssetItem
from typing import List, Optional, Dict
from fuzzywuzzy import process

def build_asset_query(db: Session, filters: Dict):
    query = db.query(Asset)

    if filters.get("condition"):
        print(f"Filtering by condition: {filters['condition']}")
        query = query.filter(Asset.condition == filters["condition"])

    if filters.get("status_id") is not None:
        print(f"Filtering by status_id: {filters['status_id']}")
        query = query.filter(Asset.status_id == filters["status_id"])

    if filters.get("location_id") is not None:
        print(f"Filtering by location_id: {filters['location_id']}")
        query = query.filter(Asset.location_id == filters["location_id"])

    if filters.get("department_id") is not None:
        print(f"Filtering by department_id: {filters['department_id']}")
        query = query.filter(Asset.department_id == filters["department_id"])

    if filters.get("user_id") is not None:
        print(f"Filtering by user_id: {filters['user_id']}")
        query = query.filter(Asset.user_id == filters["user_id"])

    return query

#finds an asset with its tag. Pretty straightforward
def handle_asset_tag_lookup(asset_tag, db):
    asset = db.query(Asset).filter(Asset.asset_tag == asset_tag).first()
    if not asset:
        return QueryResponse(answer=f"No asset found with tag {asset_tag}")
    
    return QueryResponse(answer=f"Asset Name: {asset.name}, Tag: {asset.asset_tag}, Serial: {asset.serial}, Purchase Date: {asset.purchase_date}")

#this function takes filters and disect the database accordingly
def handle_asset_filter_query(filters: Dict, db:Session) -> QueryResponse:
    # a lazy SQLAlchemy query object. Like a teenager with promise: full of potential but hasn’t done anything yet

    print(f"Applying filters: {filters}")
    query = build_asset_query(db, filters)
    #with the constructed query object we finally execute 
    assets = query.all()
    print(f"Query returned {len(assets)} assets")
    
    if not assets:
        return QueryResponse(answer="No assets found matching your criteria.")
    
    result = f"Found {len(assets)} assets matching your criteria."

    # Build response list
    asset_items = []
    for asset in assets:
        status_obj = db.query(Status).get(asset.status_id) if asset.status_id else None
        location_obj = db.query(Location).get(asset.location_id) if asset.location_id else None
        department_obj = db.query(Department).get(asset.department_id) if asset.department_id else None

        # we havent fucked with the users for now 
        # user_obj = db.query(User).get(asset.user_id) if asset.user_id else None
        
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
    
    print("called handle_asset_filter_query")

    return QueryResponse(answer=result, assets=asset_items)

def handle_asset_count_query(filters: Dict, db: Session) -> QueryResponse:

    query = build_asset_query(db, filters)
    #Final count
    count = query.count()

    # Human-readable filter info (for debugging or client display)
    readable = [
        f"{key}: {value}"
        for key, value in filters.items()
        if value not in [None, False, "", 0]  # 0 excluded if not valid ID
    ]
    filter_desc = ", ".join(readable)
    # filter_desc = ", ".join([f"{k}: {v}" for k, v in filters.items() if v is not True])

    if filter_desc:
        return QueryResponse(answer=f"Found {count} assets matching criteria: {filter_desc}")
    else:
        return QueryResponse(answer=f"Total number of assets: {count}")# LLM agent v1
    
def handle_asset_checkout(filters: Dict, context: Dict, db: Session) -> QueryResponse:
    """
    This here function done built the /assets/<id>/checkout URL for the requested asset.

    Priority order for figuring out which asset to check out:
    1.  filters["asset_id"]
    2.  filters["asset_tag"]
    3.  context["last_asset_id"]   ← value I stashed after a 1-asset result
    """
    asset = None

    #explicit id from filters
    if "asset_id" in filters:
        asset = db.query(Asset).get(filters["asset_id"])

    #explicit tag from filters
    elif "asset_tag" in filters:
        asset = db.query(Asset).filter(
            Asset.asset_tag == filters["asset_tag"]
        ).first()

    #single-asset follow-up (what we saved in the context)
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

def suggest_similar_location(name: str, all_names: List[str], threshold=70, limit=3):
    return [n for n, s in process.extract(name, all_names, limit=limit) if s >= threshold]