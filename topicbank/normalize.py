from datetime import datetime

def normalize_item(raw, category_hint=None, expl=0, tags=None):
    return {
        "id": raw["id"],
        "text": raw["text"].strip(),
        "source": raw["source"],
        "category": category_hint or "unknown",
        "explicit_level": expl,
        "tags": tags or [],
        "language": "en",
        "created_at": datetime.utcnow().isoformat() + "Z"
    }
