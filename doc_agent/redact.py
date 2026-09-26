import re

PATTERNS = [
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"), "API_KEY=<redacted>"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"), "API_KEY=<redacted>"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AWS_ACCESS_KEY_ID=<redacted>"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"), "GITHUB_TOKEN=<redacted>"),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}\b"), "GOOGLE_API_KEY=<redacted>"),
    (re.compile(r"\bxox[abpr]-[A-Za-z0-9\-]{10,}\b"), "SLACK_TOKEN=<redacted>"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----"), "PRIVATE_KEY=<redacted>"),
    (re.compile(r"(?i)\b(api[_-]?key|secret|password|passwd|token|auth)\b(\s*[:=]\s*)(['\"]?)[^\s'\"]{6,}\3"),
     r"\1\2<redacted>"),
]


def redact(text):
    hits = 0
    for pat, repl in PATTERNS:
        text, n = pat.subn(repl, text)
        hits += n
    return text, hits
