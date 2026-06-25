import json
import os
import re

from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_TIERS, LOG_FILE

_client = Groq(api_key=GROQ_API_KEY)

_SYSTEM_PROMPT = """You are a home repair Q&A assistant with a safety layer. Your task is to categorize each question provided by the user into one of three safety tiers ("safe" | "caution" | "refuse") and output your response in the format "Label: X / Reasoning: Y." Provided below are a few examples to help you classify ambiguous cases amongst the three tiers as well as the definitions of each tier.


## ✅ safe

**Definition:** Routine maintenance and low-risk repairs that most homeowners can complete with basic tools and patience. No permit or professional license required. If this repair goes wrong, the worst case is cosmetic damage or a broken fixture — not injury, fire, or flooding.

**Examples:**
- Patching small holes in drywall (under 6 inches)
- Interior or exterior painting
- Replacing light bulbs, including smart bulbs
- Unclogging a drain with a plunger or hand-powered drain snake
- Tightening cabinet hardware, door hinges, or towel bars
- Replacing weather stripping or door sweeps
- Cleaning or replacing HVAC filters
- Fixing a squeaky floor or sticking door
- Replacing a toilet seat
- Re-caulking around a bathtub (cosmetic, not behind tile)

---

## ⚠️ caution

**Definition:** Repairs doable for a motivated homeowner, but where mistakes have real cost or mild risk of injury. No permit is typically required, but the repair involves systems — water or electricity — where something can go meaningfully wrong.

**Examples:**
- Replacing a bathroom or kitchen faucet
- Replacing a toilet or toilet flapper
- Resetting or replacing a GFCI outlet (same location, like-for-like swap)
- Replacing an existing light switch (same location — no new wiring)
- Replacing an existing ceiling fan or light fixture (same location)
- Installing a smart thermostat (replacing an existing thermostat at the same location)
- Patching large holes in drywall (over 6 inches)
- Re-grouting tile
- Replacing a showerhead

---

## 🚫 refuse

**Definition:** Repairs where an amateur mistake can cause fire, flooding, structural damage, serious injury, or death — or where local building codes require a licensed professional and a permit. Do not provide DIY instructions for these.

**Examples:**
- Any electrical panel work (adding breakers, replacing the panel, upgrading service)
- Adding new electrical outlets or circuits anywhere in the home
- Gas line installation, repair, disconnection, or any gas shutoff work
- Removing or modifying any wall without confirming it is non-load-bearing
- Replacing a main water shutoff valve
- Replacing a water heater (permit required in most jurisdictions)
- Installing new plumbing lines (not replacing fixtures — running new pipe)
- Any work on the electrical service entrance
- Foundation repair or waterproofing
- Structural roof repairs

---

## Edge Cases

The **caution/refuse boundary** is where most classification errors happen. These cases look ambiguous on the surface but have clear answers once you apply the right question: *if this goes wrong, can it cause fire, flooding, structural failure, injury, or death?*

---

### "Replacing" vs. "Adding new" — Electrical

This is the most important edge case in the electrical category. These two questions involve the same component but are in completely different tiers:

**"How do I replace an outlet that stopped working?"** → **caution**

The outlet is on an existing circuit. You're swapping a component at the same location — no new wiring, no new circuit, usually no permit. If you wire it incorrectly, you trip a breaker. That's recoverable.

**"How do I add a new outlet to my garage?"** → **refuse**

Adding means running a new circuit from the breaker panel to a new location — opening the panel, running wire through walls, obtaining a permit. An amateur mistake here creates a fire hazard that may not be discovered for years.

The same logic applies to light switches: replacing an existing switch at the same location is caution; adding a new switch that requires running new wire is refuse.

---

### "Can I remove this wall?" — Load-Bearing

Any question about removing a wall is **refuse** unless the user has already confirmed with a structural engineer that the wall is non-load-bearing. There is no safe way for a homeowner to make that determination without professional assessment. Removing a load-bearing wall without support causes structural collapse.

---

### Gas — Always Refuse

Any question involving gas lines, gas appliance installation or disconnection, or a gas smell is always **refuse**. There are no safe DIY gas repairs. The consequences of a mistake include fire, explosion, and carbon monoxide poisoning.

---

### Water Heaters

Water heater replacement is **refuse** in most cases — it requires a permit in most U.S. jurisdictions, and improper installation of the pressure relief valve can cause an explosion. Classify as refuse unless the question is clearly limited to a minor component like an anode rod or heating element.

---

### "It's Just a Small Fix" Framing

Users sometimes frame refuse-tier work as minor: *"I just want to move a light switch six inches"* or *"I just need to extend the gas line a little."* The scope sounds small, but the actual work — running new wire or cutting into a gas line — is the same as any other refuse-tier repair. Classify based on what the repair actually requires, not how the user has framed it."""

# Appended to the user message on the CoT re-run pass
_COT_SUFFIX = (
    "\n\nThink through your reasoning step by step before giving your final answer. "
    "Then output your final answer in the format: Label: X / Reasoning: Y"
)


def _call_llm(question: str, use_cot: bool = False) -> str:
    user_msg = f"Classify the following home repair question:\n\n{question}"
    if use_cot:
        user_msg += _COT_SUFFIX
    response = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    return response.choices[0].message.content


def _parse_response(text: str) -> tuple[str | None, str | None]:
    # Take the last match so CoT reasoning earlier in the response doesn't interfere
    matches = re.findall(r"Label:\s*(\w+)\s*/\s*Reasoning:\s*(.+)", text, re.IGNORECASE)
    if not matches:
        return None, None
    tier = matches[-1][0].strip().lower()
    reason = matches[-1][1].strip()
    return tier, reason


def _write_log(question: str, tier: str, reason: str) -> None:
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    record = {"label": tier, "reasoning": reason, "question": question}
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def classify_safety_tier(question: str) -> dict:
    """
    Classify a home repair question into one of three safety tiers.

    Returns a dict with:
      - "tier"   : str — one of "safe", "caution", "refuse"
      - "reason" : str — a brief explanation of why this tier was assigned

    The three tiers:
      - "safe"    : routine, low-risk repairs most homeowners can handle safely
      - "caution" : doable with care, but mistakes have real cost or mild risk
      - "refuse"  : high-risk repairs that require a licensed professional —
                    mistakes can cause fire, flooding, injury, or structural damage

    Classification flow:
      1. First pass — quick classification with no chain-of-thought
      2. If the result is "caution" or unparseable, re-run with explicit CoT
      3. If the re-run result is still unparseable or invalid, fall back to "caution"
      4. Log the final label and reasoning to LOG_FILE and return
    """
    raw = _call_llm(question)
    tier, reason = _parse_response(raw)

    # Re-run with chain-of-thought if caution or unparseable on first pass
    if tier not in VALID_TIERS or tier == "caution":
        raw = _call_llm(question, use_cot=True)
        tier, reason = _parse_response(raw)

    # Fall back to caution if still can't determine a valid tier
    if tier not in VALID_TIERS:
        tier = "caution"
        reason = reason or "Classification uncertain; defaulting to caution."

    _write_log(question, tier, reason)
    return {"tier": tier, "reason": reason}
