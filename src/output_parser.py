"""Preserve the notebook parser, with a narrow unescaped-quote fallback."""
import json
import re

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser

_PARSER = JsonOutputParser()
_FORWARD = re.compile(
    r'\s*\{\s*"status"\s*:\s*"(?P<status>valid|out_of_scope)"\s*,\s*'
    r'"response"\s*:\s*"(?P<body>[\s\S]*)"\s*\}\s*'
)
_REVERSE = re.compile(
    r'\s*\{\s*"response"\s*:\s*"(?P<body>[\s\S]*)"\s*,\s*'
    r'"status"\s*:\s*"(?P<status>valid|out_of_scope)"\s*\}\s*'
)


def _escape_inner_quotes(body):
    repaired = []
    escaped = False
    for position, character in enumerate(body):
        if escaped:
            repaired.append(character)
            escaped = False
        elif character == "\\":
            repaired.append(character)
            escaped = True
        elif character == '"':
            # Refuse ambiguous field boundaries rather than turning extra keys
            # or concatenated objects into message text.
            tail = body[position + 1:]
            if re.match(r'\s*,\s*"[^"\r\n]+"\s*:', tail) or re.match(r'\s*\}\s*\{', tail):
                raise ValueError("Ambiguous JSON field boundary")
            repaired.append('\\"')
        else:
            repaired.append(character)
    # Decode existing JSON escapes as usual; invalid escapes remain errors.
    return json.loads('"' + ''.join(repaired) + '"')


def parse_qwen_output(text):
    try:
        return _PARSER.invoke(text)
    except OutputParserException as original_error:
        source = text.strip()
        fenced = re.fullmatch(r'```(?:json)?\s*([\s\S]*?)\s*```', source, re.IGNORECASE)
        if fenced:
            source = fenced.group(1)
        match = _FORWARD.fullmatch(source) or _REVERSE.fullmatch(source)
        if match is None:
            raise
        try:
            response = _escape_inner_quotes(match.group("body"))
        except (ValueError, json.JSONDecodeError):
            raise original_error from None
        return {"status": match.group("status"), "response": response}
