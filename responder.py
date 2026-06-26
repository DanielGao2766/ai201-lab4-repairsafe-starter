from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL

_client = Groq(api_key=GROQ_API_KEY)

_SYSTEM_PROMPTS = {
    "safe": (
        "You are a helpful home repair assistant. Answer the user's question with clear, "
        "specific, and actionable instructions. Include what tools and materials they'll need, "
        "step-by-step guidance, and any practical tips that will help them succeed."
    ),
    "caution": (
        "You are a home repair assistant. The repair the user is asking about carries a real "
        "risk of injury or property damage if done incorrectly. Start your response by clearly "
        "stating this risk and explaining specifically why it exists (e.g., working with live "
        "electrical components, risk of pipe flooding). Then provide step-by-step instructions. "
        "Throughout, recommend that the user hire a licensed professional if they feel uncertain "
        "at any point. Do not downplay the risks."
    ),
    "refuse": (
        "Do NOT provide any how-to instructions, steps, procedures, methods, or guidance — "
        "not even general, partial, or framed as \"what a professional would do.\" Do not "
        "describe the technical process in any way.\n\n"
        "Your response must do exactly three things:\n"
        "1. Tell the user clearly that this repair requires a licensed professional.\n"
        "2. Explain briefly why it is dangerous (e.g., fire risk, permit required, risk of "
        "structural collapse).\n"
        "3. Tell them what type of professional to contact (electrician, plumber, structural "
        "engineer, etc.).\n\n"
        "If the user pushes back or asks for partial guidance, hold firm. Do not accommodate "
        "requests for \"just a tip\" or \"just an overview.\""
    ),
}


def generate_safe_response(question: str, tier: str) -> str:
    """
    Generate a response to a home repair question, calibrated to its safety tier.

    Returns the response as a plain string.

    Tier behavior:
      - "safe"    : answer helpfully and directly
      - "caution" : answer with explicit safety warnings and professional recommendation
      - "refuse"  : decline to give instructions; explain the danger and refer to a professional
      - unknown   : treated as "caution" to fail safe rather than fail open
    """
    system_prompt = _SYSTEM_PROMPTS.get(tier, _SYSTEM_PROMPTS["caution"])

    response = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content
