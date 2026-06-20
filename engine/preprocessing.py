from __future__ import annotations

import re

MIN_FRAGMENT_TOKENS = 3

STOPWORDS = {
    "public", "private", "protected", "void", "new", "null", "int", "string",
    "static", "final", "return", "class", "this", "super", "true", "false",
    "boolean", "long", "double", "float", "byte", "char", "short", "object",
    "list", "map", "set", "if", "else", "for", "while", "do", "try", "catch",
    "finally", "throw", "throws", "import", "package", "extends", "implements",
    "instanceof", "interface", "abstract", "synchronized", "volatile",
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "of",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "it", "its", "with", "for", "not", "by", "from", "as", "that", "this",
    "which", "who", "when", "where", "how", "all", "each", "both", "more",
    "s", "e",
}

_SUBWORD_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def split_identifier(token: str) -> list[str]:
    out: list[str] = []
    for part in re.split(r"[_\s]+", token):
        out.extend(_SUBWORD_RE.findall(part))
    return [w.lower() for w in out if w]


def tokenize(text: str) -> list[str]:
    """Sub-tokenize identifiers, lowercase, and drop stopwords."""
    toks: list[str] = []
    for raw in text.split():
        for sub in split_identifier(raw):
            if sub and sub not in STOPWORDS:
                toks.append(sub)
    return toks


_METHOD_HEAD = re.compile(
    r"(?m)^[ \t]*"
    r"(?:@(?:\w+\.)*\w+(?:\([^)]*\))?\s+)*"
    r"(?:(?:public|private|protected)\s+)?"
    r"(?:static\s+)?(?:final\s+)?(?:synchronized\s+)?"
    r"(?:<[^>]+>\s+)?"
    r"[\w\[\]<>,.\s?]+\s+"
    r"(\w+)\s*"
    r"\(",
)

_NON_METHOD_NAMES = frozenset({
    "if", "for", "while", "switch", "catch", "do", "try", "else", "return", "new",
})


def split_java_methods(code: str) -> list[dict[str, str]]:
    source = code.strip()
    if not source:
        return []

    spans: list[tuple[int, int, str]] = []

    for match in _METHOD_HEAD.finditer(source):
        name = match.group(1)
        if name in _NON_METHOD_NAMES:
            continue

        start = match.start()
        brace_start = source.find("{", match.end())
        if brace_start == -1:
            continue

        header = source[match.end():brace_start]
        if ";" in header.split("//", 1)[0]:
            continue

        depth = 0
        end = brace_start
        for idx in range(brace_start, len(source)):
            ch = source[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = idx + 1
                    break
        else:
            continue

        if any(start < existing_end and end > existing_start for existing_start, existing_end, _ in spans):
            continue

        spans.append((start, end, name))

    spans.sort(key=lambda item: item[0])

    methods = [
        {"name": name, "code": source[start:end].strip()}
        for start, end, name in spans
    ]

    if methods:
        return methods

    return [{"name": "(entire file)", "code": source}]


def format_method_summaries(parts: list[tuple[str, str]]) -> str:
    lines: list[str] = []
    for name, text in parts:
        cleaned = text.strip()
        if cleaned:
            lines.append(f"• {name}: {cleaned}")
        else:
            lines.append(f"• {name}: (no output)")
    return "\n".join(lines)


def split_code_statements(code: str) -> list[str]:
    parts = re.split(r"[;{}\n]", code)
    frags = [re.sub(r"\s+", " ", p).strip() for p in parts]
    frags = [f for f in frags if f]

    merged: list[str] = []
    for frag in frags:
        if len(frag.split()) < MIN_FRAGMENT_TOKENS and merged:
            merged[-1] += " " + frag
        else:
            merged.append(frag)

    return merged if merged else [re.sub(r"\s+", " ", code).strip()]
