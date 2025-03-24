import re

def extract_asset_tag(question: str) -> str:
    # Simple regex to find asset tags (e.g., "asset 12345")
    match = re.search(r'asset\s+(\w+)', question, re.IGNORECASE)
    if match:
        return match.group(1)
    return None
