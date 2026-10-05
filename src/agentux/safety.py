"""Destructive-command detection.

This is deliberately a small, extensible *prompt* for the user -- not a
security boundary. A blacklist cannot be complete: commands can be obfuscated,
generated at runtime, aliased, or run through another interpreter. The real
protection is that nothing executes without the user seeing it.

To extend it, add a ``(regex, reason)`` pair to ``_RULES`` or register your own
callable with :func:`add_rule`.
"""

from __future__ import annotations

import re
from typing import Callable, List, Pattern, Tuple

#: A rule maps a pattern to the reason shown to the user.
Rule = Tuple[Pattern[str], str]
_RULES: List[Rule] = []


def _add(pattern: str, reason: str) -> None:
    _RULES.append((re.compile(pattern, re.IGNORECASE), reason))


# Destructive filesystem operations. Recursive flags are called out
# separately from plain force-delete so the reason shown to the user is
# accurate rather than alarmist.
_add(r"\brm\s+-[a-z]*r[a-z]*\b", "recursive delete")
_add(r"\brm\s+-[a-z]*f[a-z]*\b", "force delete")
_add(r"\brmdir\b.*-", "recursive directory delete")
_add(r"\bfind\b.*-delete", "mass file deletion")
_add(r"\btruncate\b.*-s\s*0", "truncating a file to zero bytes")

# Disk and filesystem formatting.
_add(r"\bmkfs(\.\w+)?\b", "filesystem format")
_add(r"\bdd\b[^|;]*\bof=/dev/", "raw disk write")
_add(r"\bdiskutil\b[^|;]*\b(erase|reformat|partition)", "disk erase/repartition")
_add(r">\s*/dev/[sh]d[a-z]", "raw write to a block device")

# Process and machine control.
_add(r"\bsudo\b", "privilege escalation")
_add(r"\b(shutdown|reboot|halt|poweroff)\b", "system shutdown")
_add(r"\bkillall\s+-9\b", "killing all processes")
_add(r"\bpkill\b.*-9", "killing processes by pattern")

# Git history and remote operations that lose work.
_add(r"\bgit\s+reset\s+--hard\b", "discards uncommitted changes")
_add(r"\bgit\s+clean\b[^|;]*-[a-z]*[fd]", "deletes untracked files")
_add(r"\bgit\s+push\b[^|;]*(--force|-f)\b", "force-push overwrites remote history")
_add(r"\bgit\s+branch\b[^|;]*\s-D\b", "force-deletes a branch")

# Credential and secret destruction.
_add(r"\brm\b[^|;]*(\.ssh|\.aws|\.gnupg|\.kube|id_rsa|credentials)", "credential destruction")
_add(r"\bsecurity\s+delete-(generic|internet)-password\b", "keychain deletion")
_add(r"\bhistory\s+-c\b", "erasing shell history")
_add(r"\b(shred|srm)\b", "secure file erasure")

# Obvious fork bombs.
_add(r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", "fork bomb")

# Obfuscation used to slip destructive commands past a naive scan.
_add(r"(ba|z|k|)sh\s+-c\s+['\"].*\brm\b", "obfuscated delete")

# Piping downloaded content straight into a shell interpreter.
_add(r"\b(curl|wget|fetch)\b[^|]*\|\s*(sudo\s+)?(ba|z|k|)?sh\b", "piping a download into a shell")
_add(r"\b(curl|wget)\b[^|;]*\|\s*(sudo\s+)?(python3?|perl|ruby)\b", "piping a download into an interpreter")


def add_rule(pattern: str, reason: str) -> None:
    """Register an extra destructive-command pattern."""
    _add(pattern, reason)


def add_check(fn: Callable[[str], bool], reason: str = "custom safety check") -> None:
    """Register an arbitrary predicate for a destructive command."""
    _RULES.append((re.compile(r"(?!)"), reason))  # never matches on its own
    _CHECKS.append((fn, reason))


#: Predicates added via :func:`add_check`.
_CHECKS: List[Tuple[Callable[[str], bool], str]] = []


def scan(command: str) -> List[str]:
    """Return the reasons ``command`` looks destructive (empty if it looks safe).

    A command is scanned in full, not token by token, so a destructive command
    hidden inside an ``&&`` chain or a subshell is still caught.
    """
    reasons: List[str] = []
    for pattern, reason in _RULES:
        if pattern.search(command) and reason not in reasons:
            reasons.append(reason)
    for check, reason in _CHECKS:
        try:
            if check(command) and reason not in reasons:
                reasons.append(reason)
        except Exception:
            # A broken custom check must not silently allow a command through.
            if "custom safety check failed" not in reasons:
                reasons.append("custom safety check failed")
    return reasons


def is_destructive(command: str) -> bool:
    """Whether ``command`` requires explicit confirmation before running."""
    return bool(scan(command))


__all__ = ["scan", "is_destructive", "add_rule", "add_check"]
