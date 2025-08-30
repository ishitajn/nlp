"""
This module holds constant values and mappings used across the application.
"""

# --- Mappings for Frontend Alignment ---

DATE_ARC_PHASE_MAP = {
    "Icebreaker": "OPENER",
    "Rapport Building": "VIBING",
    "Escalation": "ESCALATION",
    "Logistics": "PLANNING",
    "Explicit Banter": "SEXUAL", # Assuming this maps to a more general category if not explicit
    # Default mappings for unhandled phases
    "default": "VIBING"
}

INTENT_MAP = {
    "Gathering Information": "QUESTIONING",
    "Making Plans": "PLANNING",
    "Building Comfort": "REACTING_TO_HUMOR", # This is an approximation
    "Testing Boundaries": "FLIRTING_OR_SEXUAL",
    "Expressing Desire": "FLIRTING_OR_SEXUAL",
    # Storytelling is not explicitly detected yet, but can be added
    "Storytelling": "STORYTELLING"
}
