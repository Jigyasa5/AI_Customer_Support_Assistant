# RAG Guardrails

# Common prompt-injection phrases
INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all previous instructions",
    "ignore the instructions",
    "forget your instructions",
    "disregard previous instructions",
    "disregard the instructions",
    "you are now",
    "act as",
    "system prompt",
    "reveal your prompt",
    "show me your prompt",
    "developer message",
    "jailbreak",
]


# Detect prompt injection

def detect_prompt_injection(query):

    query_lower = query.lower()

    for pattern in INJECTION_PATTERNS:

        if pattern in query_lower:
            return True

    return False


#----------------- Validate user query --------------------

def validate_query(query):

    if not query or not query.strip():

        return {
            "allowed": False,
            "reason": "Empty query."
        }

    if detect_prompt_injection(query):

        return {
            "allowed": False,
            "reason": "Potential prompt injection detected."
        }

    return {
        "allowed": True,
        "reason": "Query accepted."
    }


#---------------- Safe query handler -----------------


def get_safe_query(query):

    validation = validate_query(query)

    if not validation["allowed"]:

        return None, validation["reason"]

    return query.strip(), None