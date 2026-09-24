"""
SkyGuard — Step 6: Plain-English Operational Insight Generator

This module exposes a single public function:

    generate_route_briefing(route_facts: RouteFacts) -> str

It builds the exact prompt specified in the SkyGuard brief, sends it to the
Gemini API (model: gemini-1.5-flash), and returns the three-sentence briefing.

SECURITY
  The Gemini API key is read exclusively from the environment variable
  GEMINI_API_KEY — it is never hardcoded or logged.

DEPENDENCIES
  pip install google-generativeai

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

_GEMINI_MODEL_NAME = "gemini-1.5-flash"   # fast, low-cost; swap for gemini-1.5-pro
                                           # if richer reasoning is needed


def _get_gemini_client():
    """
    Build a google.genai Client using the API key from the environment.

    Raises
    ------
    EnvironmentError
        If GEMINI_API_KEY is not set — fails loudly rather than silently
        sending an unauthenticated request.
    """
    from google import genai  # lazy — only needed when AI page is actually used

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set.\n"
            "Set it with:  export GEMINI_API_KEY='your-key-here'  (Linux/Mac)\n"
            "          or: $env:GEMINI_API_KEY='your-key-here'    (PowerShell)\n"
            "          or add it to your Colab Secrets panel."
        )
    return genai.Client(api_key=api_key)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_route_briefing(
    facts       : RouteFacts,
    model_name  : str          = _GEMINI_MODEL_NAME,
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
        Gemini model to use (default: gemini-1.5-flash).
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
        If GEMINI_API_KEY is not set.
    google.api_core.exceptions.GoogleAPIError
        Propagated as-is if the API call fails (quota, network, etc.).
    """
    from google.genai import types as genai_types  # lazy — mirrors _get_gemini_client

    prompt  = _build_prompt(facts)
    client  = _get_gemini_client()

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature       = temperature,
            max_output_tokens = max_tokens,
        ),
    )

    # .text raises if the response was blocked by safety filters — the
    # caller will see a clear exception rather than an empty string.
    return response.text.strip()


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
