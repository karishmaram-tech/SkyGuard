"""
SkyGuard — Step 6: Plain-English Operational Insight Generator

This module exposes a single public function:

    generate_route_briefing(route_facts: RouteFacts) -> str

It builds the exact prompt specified in the SkyGuard brief, sends it to the
Groq API (model: llama-3.1-8b-instant), and returns the three-sentence briefing.

SECURITY
  The Groq API key is read exclusively from the environment variable
  GROQ_API_KEY — it is never hardcoded or logged.

DEPENDENCIES
  pip install groq

USAGE EXAMPLE
  from insights import generate_route_briefing, RouteFacts

  facts = RouteFacts(
      route              = "ORD → LAX",
      expected_risk_pct  = 34.2,
      actual_rate_pct    = 41.7,
      top_cause          = "Late Aircraft",
      top_cause_pct      = 52.3,
      sample_size        = 1840,
  )
  print(generate_route_briefing(facts))
"""

import os
from dataclasses import dataclass
from typing import Optional

# google-genai is imported lazily inside _get_gemini_client() so that importing
# this module never fails in environments where the package is not installed
# (e.g. Streamlit Cloud before requirements are fully resolved, or any context
# where the AI page is simply not used).


# ─────────────────────────────────────────────────────────────────────────────
# Data structure: all facts the prompt template requires
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RouteFacts:
    """
    Structured container for the route facts fed into the prompt.

    Fields
    ------
    route              : Human-readable route label, e.g. "ORD → LAX"
    expected_risk_pct  : Model's predicted delay risk for this route (0–100).
    actual_rate_pct    : Observed delay rate in the last 30 days (0–100).
    top_cause          : Name of the primary delay cause, e.g. "Late Aircraft".
    top_cause_pct      : Percentage of delays attributable to that cause (0–100).
    sample_size        : Number of flights the statistics are based on.
    """
    route             : str
    expected_risk_pct : float
    actual_rate_pct   : float
    top_cause         : str
    top_cause_pct     : float
    sample_size       : int


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder — exact template mandated by the project spec
# ─────────────────────────────────────────────────────────────────────────────

_PROMPT_TEMPLATE = (
    "You are an airline operations analyst. "
    "Given these facts about Route {route}: "
    "expected on-time risk {expected_risk_pct:.1f}%, "
    "actual last-30-day rate {actual_rate_pct:.1f}%, "
    "primary delay cause {top_cause} ({top_cause_pct:.1f}% of delays), "
    "sample size {sample_size} flights — "
    "write a 3-sentence briefing: "
    "(1) state the fact, "
    "(2) explain the likely driver, "
    "(3) recommend one concrete operational action. "
    "Do not invent numbers not provided above."
)


def _build_prompt(facts: RouteFacts) -> str:
    """
    Populate the fixed template with the provided RouteFacts.
    Returns the fully-formed prompt string.
    """
    return _PROMPT_TEMPLATE.format(
        route             = facts.route,
        expected_risk_pct = facts.expected_risk_pct,
        actual_rate_pct   = facts.actual_rate_pct,
        top_cause         = facts.top_cause,
        top_cause_pct     = facts.top_cause_pct,
        sample_size       = facts.sample_size,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Gemini client — initialised lazily so import doesn't fail without the key
# ─────────────────────────────────────────────────────────────────────────────

_GROQ_MODEL_NAME = "llama-3.1-8b-instant"  # fast, free tier; swap for llama-3.3-70b-versatile
                                            # for higher quality reasoning


def _get_groq_client():
    """
    Build a Groq client using the API key from the environment.

    Raises
    ------
    EnvironmentError
        If GROQ_API_KEY is not set.
    """
    from groq import Groq  # lazy — only needed when AI page is actually used

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY environment variable is not set.\n"
            "Get a free key at https://console.groq.com and set it with:\n"
            "  export GROQ_API_KEY='gsk_...'   (Linux/Mac)\n"
            "  $env:GROQ_API_KEY='gsk_...'     (PowerShell)"
        )
    return Groq(api_key=api_key)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_route_briefing(
    facts       : RouteFacts,
    model_name  : str          = _GROQ_MODEL_NAME,
    temperature : float        = 0.3,    # low temperature → factual, consistent tone
    max_tokens  : Optional[int] = 256,   # three sentences fit well within 256 tokens
) -> str:
    """
    Generate a plain-English three-sentence operational briefing for a route.

    Parameters
    ----------
    facts       : RouteFacts
        The structured facts to embed in the prompt.
    model_name  : str
        Groq model to use (default: llama-3.1-8b-instant).
    temperature : float
        Controls response creativity. 0.3 keeps the output factual and
        consistent across repeated calls on the same input.
    max_tokens  : int | None
        Approximate upper bound on the response length.

    Returns
    -------
    str
        The generated briefing text, stripped of leading/trailing whitespace.

    Raises
    ------
    EnvironmentError
        If GROQ_API_KEY is not set.
    groq.APIError
        Propagated as-is if the API call fails (quota, network, etc.).
    """
    prompt   = _build_prompt(facts)
    client   = _get_groq_client()
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Batch helper — generate briefings for a list of routes
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_briefings(route_facts_list: list[RouteFacts]) -> list[dict]:
    """
    Call generate_route_briefing for every route in the list.

    Returns a list of dicts:
        [{"route": "ORD → LAX", "briefing": "..."}, ...]

    Errors for individual routes are caught and stored as the briefing text
    so a single bad API response doesn't abort the whole batch.
    """
    results = []
    for facts in route_facts_list:
        try:
            briefing = generate_route_briefing(facts)
        except Exception as exc:
            briefing = f"[ERROR generating briefing: {exc}]"
        results.append({"route": facts.route, "briefing": briefing})
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Self-test  (run directly: python insights.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Replace these values with real computed facts from model.py /
    # anomalous_routes.csv before running.
    test_facts = RouteFacts(
        route              = "ORD → LAX",
        expected_risk_pct  = 34.2,
        actual_rate_pct    = 41.7,
        top_cause          = "Late Aircraft",
        top_cause_pct      = 52.3,
        sample_size        = 1840,
    )

    print("Prompt that will be sent to Gemini:")
    print("─" * 60)
    print(_build_prompt(test_facts))
    print("─" * 60)
    print()

    print("Calling Gemini API …")
    briefing = generate_route_briefing(test_facts)
    print()
    print("Generated briefing:")
    print(briefing)
