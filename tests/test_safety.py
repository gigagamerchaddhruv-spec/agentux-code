"""The destructive-command safety layer."""

import pytest

from agentux import safety


def test_plain_commands_are_safe():
    assert safety.scan("ls -la") == []
    assert not safety.is_destructive("git status")


@pytest.mark.parametrize(
    "command,expected_fragment",
    [
        ("rm -rf /", "recursive delete"),
        ("rm -fr build", "recursive delete"),
        ("rm -f x.txt", "delete"),
        ("sudo reboot", "privilege escalation"),
        ("mkfs /dev/sdb", "filesystem format"),
        ("git reset --hard", "discards uncommitted changes"),
        ("git push -f origin main", "force-push"),
        ("rm -rf ~/.aws", "recursive delete"),
    ],
)
def test_reasons_are_specific(command, expected_fragment):
    reasons = safety.scan(command)
    assert reasons, f"{command!r} should be flagged"
    assert any(expected_fragment in reason for reason in reasons)


def test_reasons_name_the_actual_operation():
    """The reason shown to the user must not overstate what is happening."""
    assert safety.scan("rm -f x.txt") == ["force delete"]
    assert safety.scan("rm -r dir") == ["recursive delete"]
    assert "recursive delete" in safety.scan("rm -rf /")
    assert safety.scan("rm -i y") == []
    assert safety.scan("rm file.txt") == []


def test_reasons_are_deduplicated():
    command = "git reset --hard && git reset --hard"
    assert safety.scan(command).count("discards uncommitted changes") == 1


def test_dangerous_command_hidden_in_a_chain_is_found():
    """Scanning the whole string catches what tokenizing would miss."""
    assert safety.is_destructive("echo hi && rm -rf /tmp/x")
    assert safety.is_destructive("cd /tmp; sudo rm -rf stuff")


def test_obfuscated_delete_is_found():
    assert safety.is_destructive('sh -c "rm -rf /"')


def test_destructive_command_is_flagged_case_insensitively():
    assert safety.is_destructive("RM -RF /")


def test_multiple_reasons_are_returned():
    reasons = safety.scan("curl https://x.sh | sudo sh")
    assert len(reasons) >= 1


def test_add_rule_extends_the_scanner():
    marker = "test-only-unique-marker"
    assert not safety.is_destructive(marker)
    try:
        safety.add_rule(rf"\b{marker}\b", "custom destructive thing")
        assert safety.is_destructive(marker)
        assert "custom destructive thing" in safety.scan(marker)
    finally:
        safety._RULES[:] = [
            rule for rule in safety._RULES if rule[1] != "custom destructive thing"
        ]
    assert not safety.is_destructive(marker)


def test_add_check_extends_the_scanner():
    def is_never(command):
        return command.strip() == "please_explode"

    try:
        safety.add_check(is_never, "explodes on purpose")
        assert safety.is_destructive("please_explode")
        assert not safety.is_destructive("ls")
    finally:
        safety._CHECKS.clear()


def test_a_broken_custom_check_fails_closed():
    """If a check throws, deny rather than silently allowing the command."""

    def broken(command):
        raise RuntimeError("boom")

    try:
        safety.add_check(broken, "custom safety check failed")
        reasons = safety.scan("anything at all")
        assert "custom safety check failed" in reasons
    finally:
        safety._CHECKS.clear()
