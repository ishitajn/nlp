from typing import Dict, List

def enforce_constraints(suggestions: Dict[str, List[str]], max_items: int = 2) -> Dict[str, List[str]]:
    """
    Enforces constraints on the generated suggestions.
    - Removes duplicate suggestions.
    - Caps the number of suggestions per category to `max_items`.
    """
    constrained_suggestions = {}
    for category, items in suggestions.items():
        # Remove duplicates while preserving order
        unique_items = list(dict.fromkeys(items))

        # Cap the number of items
        constrained_suggestions[category] = unique_items[:max_items]

    return constrained_suggestions
