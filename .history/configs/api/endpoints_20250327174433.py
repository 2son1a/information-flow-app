"""
API endpoint configurations and settings.
"""

# API endpoints
ENDPOINTS = {
    "health": "/health",
    "process_text": "/process_text",
}

# API settings
API_SETTINGS = {
    "base_url": "http://localhost:8000",
    "timeout": 30,  # seconds
    "health_check_timeout": 10,  # seconds
}

# Request/Response schemas
REQUEST_SCHEMAS = {
    "process_text": {
        "text": str,
        "model": str,
    }
}

# Response schemas
RESPONSE_SCHEMAS = {
    "health": {
        "status": str,
        "version": str,
    },
    "process_text": {
        "numLayers": int,
        "numTokens": int,
        "numHeads": int,
        "tokens": list[str],
        "attentionPatterns": list[dict],
    }
} 