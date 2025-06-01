from pydantic import BaseModel
from typing import List, Optional, Dict

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