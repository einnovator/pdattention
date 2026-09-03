"""Regenerate the PRA Agent slash-command reference from its live registry."""

from pathlib import Path

from pra_hf.tui import AgentShell, write_command_reference


def main() -> None:
    shell = object.__new__(AgentShell)
    shell.agent = None
    shell.output = lambda _value: None
    shell.last_output = ""
    shell.__post_init__()
    target = Path("docs/site/agents/slash-commands.md")
    write_command_reference(target, shell.registry)
    print(target)


if __name__ == "__main__":
    main()
