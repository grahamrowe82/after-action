import re
from collections import Counter
from typing import Dict, List

MAX_INSIGHTS = 5
MAX_DELTAS = 3
LINE_LIMIT = 120

METRIC_RE = re.compile(r"\d+(?:\.\d+)?%|\b\d+\b")

INSIGHT_KEYWORDS = {
    "learned": 3,
    "observed": 3,
    "data shows": 3,
    "latency": 3,
    "ctr": 3,
    "retention": 3,
    "decided": 2,
    "approved": 2,
    "launched": 2,
    "caused": 2,
    "owner": 2,
    "ownership": 2,
    "legal": 1,
    "spent too long": 2,
    "debating": 2,
    "users said": 1,
    "users loved": 1,
    "users hated": 1,
    "support tickets": 1,
}

DELTA_PATTERNS = [
    re.compile(r"^(next time|from now on|going forward|for next time)", re.IGNORECASE),
    re.compile(r"^let['’]?s\b", re.IGNORECASE),
    re.compile(r"^let\s+us\b", re.IGNORECASE),
    re.compile(r"^we\s+(?:should|will|must|need to|ought to|plan to|can|could|are going to|want to|aim to)\b", re.IGNORECASE),
    re.compile(r"^we['’]?ll\b", re.IGNORECASE),
    re.compile(
        r"^(?:stop|start|change|fix|freeze|schedule|set|assign|add|create|document|monitor|automate|alert|establish|audit|reroute|calibrate|instrument|review|kickoff|prepare|train)\b",
        re.IGNORECASE,
    ),
]

RULE_WORDS = ["always", "never", "must", "cannot", "invariant", "principle"]

NEGATIVE_TOKENS = {
    "delay",
    "delayed",
    "late",
    "blocked",
    "blocking",
    "issue",
    "issues",
    "error",
    "errors",
    "bug",
    "bugs",
    "failure",
    "failed",
    "failing",
    "missing",
    "lack",
    "debate",
    "debating",
    "waste",
    "slow",
    "bottleneck",
    "quota",
    "limit",
    "limits",
    "owner",
    "ownership",
    "handoff",
    "handoffs",
}

STOPWORDS = {
    "the",
    "and",
    "with",
    "that",
    "this",
    "from",
    "into",
    "were",
    "have",
    "has",
    "will",
    "time",
    "next",
    "before",
    "after",
    "about",
    "then",
    "than",
    "when",
    "there",
    "their",
    "while",
    "without",
    "within",
    "across",
}


def sentence_split(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    raw_segments: List[str] = []
    for line in re.split(r"\n+", text):
        line = line.strip()
        if line:
            raw_segments.append(line)

    sentences: List[str] = []
    for segment in raw_segments:
        parts = re.split(r"(?<=[.!?])\s+", segment)
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences


def _simple_stem(word: str) -> str:
    word = word.lower()
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[: -len(suffix)]
    return word


def _signature(sentence: str) -> str:
    tokens = re.findall(r"[a-zA-Z]{3,}", sentence.lower())
    stems = sorted({_simple_stem(token) for token in tokens})
    return "-".join(stems)


def _failure_score(sentence: str) -> int:
    lowered = sentence.lower()
    hits = [token for token in NEGATIVE_TOKENS if token in lowered]
    score = len(hits)
    if "no " in lowered or "missing" in lowered or "lack" in lowered:
        score += 2
    if "owner" in lowered:
        score += 3
    if "legal" in lowered:
        score += 1
    if "rate limit" in lowered or "quota" in lowered:
        score += 1
    if "alert" in lowered or "rpm" in lowered:
        score += 1
    if "debate" in lowered or "debating" in lowered:
        score += 1
    return score


def _best_failure_sentence(sentences: List[str]) -> tuple[str, int]:
    best_sentence = ""
    best_score = 0
    for sentence in sentences:
        score = _failure_score(sentence)
        if score > best_score:
            best_sentence = sentence
            best_score = score
    return best_sentence, best_score


def _score_insight(sentence: str) -> int:
    score = 0
    if METRIC_RE.search(sentence):
        score += 3
    lowered = sentence.lower()
    for phrase, value in INSIGHT_KEYWORDS.items():
        if phrase in lowered:
            score += value
    if any(token in lowered for token in NEGATIVE_TOKENS):
        score += 1
    if _looks_like_delta(sentence):
        score -= 3
    return score


def _trim(text: str, limit: int = LINE_LIMIT) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def extract_insights(sentences: List[str]) -> List[str]:
    ranked = []
    for idx, sentence in enumerate(sentences):
        score = _score_insight(sentence)
        if score > 0:
            ranked.append((score, idx, sentence))
    ranked.sort(key=lambda item: (-item[0], item[1]))

    seen_signatures = set()
    insights: List[str] = []
    for _score, _idx, sentence in ranked:
        signature = _signature(sentence)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        insights.append(_trim(sentence))
        if len(insights) == MAX_INSIGHTS:
            break
    return insights


def _looks_like_delta(sentence: str) -> bool:
    text = sentence.strip()
    if not text:
        return False
    lowered = text.lower()
    if "next time" in lowered or "from now on" in lowered:
        return True
    for pattern in DELTA_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _normalize_imperative(sentence: str) -> List[str]:
    if not _looks_like_delta(sentence):
        return []

    text = sentence.strip()
    text = re.sub(r"[.!?]+$", "", text)

    prefixes = [
        r"^next time[:,\-]*\s*",
        r"^from now on[:,\-]*\s*",
        r"^going forward[:,\-]*\s*",
        r"^for next time[:,\-]*\s*",
        r"^let['’]?s\s+",
        r"^let\s+us\s+",
        r"^we\s+(?:should|will|must|need to|ought to|can|could|are going to|plan to|aim to)\s+",
        r"^we['’]?ll\s+",
    ]
    for pattern in prefixes:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    parts = re.split(r"[;•\n]+", text)
    commands: List[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        part = re.sub(r"^and\s+", "", part, flags=re.IGNORECASE)
        part = re.sub(r"^to\s+", "", part, flags=re.IGNORECASE)
        part = re.sub(r"\bplease\b", "", part, flags=re.IGNORECASE)
        part = re.sub(r"\s+", " ", part).strip()
        if not part:
            continue
        part = part[0].upper() + part[1:]
        if not part.endswith("."):
            part += "."
        command = _trim(part)
        if command not in commands:
            commands.append(command)
    return commands


def extract_deltas(sentences: List[str]) -> List[str]:
    deltas: List[str] = []
    for sentence in sentences:
        commands = _normalize_imperative(sentence)
        for command in commands:
            if command not in deltas:
                deltas.append(command)
        if len(deltas) == MAX_DELTAS:
            break
    return deltas


def theme_counter(sentences: List[str]) -> Counter:
    counter: Counter = Counter()
    for sentence in sentences:
        lowered = sentence.lower()
        if not any(token in lowered for token in NEGATIVE_TOKENS):
            continue
        for word in re.findall(r"[a-zA-Z]{3,}", lowered):
            stem = _simple_stem(word)
            if stem in STOPWORDS:
                continue
            weight = 2 if stem in NEGATIVE_TOKENS else 1
            counter[stem] += weight
    return counter


def _pick_theme_sentence(sentences: List[str], theme: str) -> str:
    for sentence in sentences:
        signature = _signature(sentence)
        if theme and theme in signature.split("-"):
            return sentence
    return ""


def _craft_rule_from_sentence(sentence: str, deltas: List[str]) -> str:
    lowered = sentence.lower()
    if "owner" in lowered:
        if "legal" in lowered:
            return "Always assign an owner with a due date for legal gating items at kickoff."
        match = re.search(r"no owner (?:on|for|in) ([^.;]+)", sentence, re.IGNORECASE)
        if match:
            target = match.group(1).strip().rstrip(",; ")
            return _trim(
                f"Always assign an owner with a due date for {target} at kickoff."
            )
        return "Always assign a clear owner with a due date at kickoff."
    if "rate limit" in lowered or "quota" in lowered:
        return "Never ship without rate-limit guardrails and alerts."
    if "alert" in lowered or "rpm" in lowered or "error" in lowered:
        return "Always wire alerts before launch to catch errors fast."
    if "debating" in lowered or "debate" in lowered or "spent too long" in lowered:
        return "Never leave critical debates open past kickoff—decide quickly."
    if "late" in lowered or "delayed" in lowered or "delay" in lowered:
        return "Always lock owners and timelines to prevent late handoffs."
    if deltas:
        return _trim(f"Always follow through: {deltas[0]}")
    return "Add a take-away that never changes."


def _craft_tweak_from_sentence(sentence: str) -> str | None:
    lowered = sentence.lower()
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", sentence)
    if "owner" in lowered and "legal" in lowered:
        return "Add 'owner+due' check to kickoff template."
    if "owner" in lowered:
        return "Add an 'owner+due' column to the kickoff checklist."
    if "rate limit" in lowered or "quota" in lowered:
        if numbers:
            return f"Add {numbers[0]} rpm rate-limit alert to monitoring."
        return "Add rate-limit guardrail checks to monitoring."
    if "alert" in lowered or "rpm" in lowered:
        if numbers:
            return f"Add {numbers[0]} rpm alert to monitoring."
        return "Wire launch alerts into monitoring."
    if "debate" in lowered or "debating" in lowered:
        return "Add a 'decision owner + deadline' prompt to kickoff prep."
    if "late" in lowered or "delayed" in lowered:
        return "Add owner+due check to block late handoffs."
    return None


def extract_irreversible_lesson(
    sentences: List[str], insights: List[str], deltas: List[str]
) -> str:
    for sentence in sentences:
        lowered = sentence.lower()
        if any(word in lowered for word in RULE_WORDS):
            return _trim(sentence)

    best_sentence, _ = _best_failure_sentence(sentences)
    if best_sentence:
        lesson = _craft_rule_from_sentence(best_sentence, deltas)
        return _trim(lesson)

    themes = theme_counter(sentences)
    if themes:
        theme, _ = themes.most_common(1)[0]
        base_sentence = _pick_theme_sentence(sentences, theme)
        if base_sentence:
            base_sentence = re.sub(r"[.!?]+$", "", base_sentence).strip()
            lesson = f"Always tackle {theme} issues early: {base_sentence}."
            return _trim(lesson)
        if deltas:
            lesson = f"Always follow through: {deltas[0]}"
            return _trim(lesson)
        return _trim(f"Always tackle {theme} risks before kickoff.")

    if deltas:
        lesson = f"Always follow through: {deltas[0]}"
        return _trim(lesson)

    return "Add a take-away that never changes."


def suggest_system_tweak(sentences: List[str], deltas: List[str]) -> str:
    best_sentence, best_score = _best_failure_sentence(sentences)
    if best_sentence:
        tweak = _craft_tweak_from_sentence(best_sentence)
        if tweak:
            return _trim(tweak)

    if deltas:
        delta = deltas[0].rstrip(".")
        tweak = f"Add '{delta}' to the team checklist."
        return _trim(tweak)

    themes = theme_counter(sentences)
    if themes:
        theme, _ = themes.most_common(1)[0]
        tweak = f"Add a guardrail for recurring {theme} friction."
        return _trim(tweak)

    return "Capture a system change to keep improving."


def assemble(text: str) -> Dict[str, List[str] | str]:
    sentences = sentence_split(text)
    insights = extract_insights(sentences)
    deltas = extract_deltas(sentences)
    lesson = extract_irreversible_lesson(sentences, insights, deltas)
    tweak = suggest_system_tweak(sentences, deltas)

    if len(insights) < MAX_INSIGHTS:
        hints = [
            "— Add more metrics or decisions for sharper insights.",
            "— Capture outcomes or data points to surface insights.",
            "— Note what users said so we can learn.",
            "— Mention measurable shifts to fill this insight.",
            "— Highlight a decision or result we can keep.",
        ]
        for hint in hints[len(insights) : MAX_INSIGHTS]:
            insights.append(hint)
    if len(deltas) < MAX_DELTAS:
        hints = [
            "— Call out what to change next time.",
            "— Note a stop/start to sharpen the retro.",
            "— Add an action we can own going forward.",
        ]
        for hint in hints[len(deltas) : MAX_DELTAS]:
            deltas.append(hint)

    if not lesson:
        lesson = "Add a take-away that never changes."
    if not tweak:
        tweak = "Capture a system change to keep improving."

    return {
        "insights": insights,
        "deltas": deltas,
        "lesson": lesson,
        "system_tweak": tweak,
    }
