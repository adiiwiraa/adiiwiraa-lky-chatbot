"""Terminal chat interface.

Run from the project root: python -m lkybot.cli
"""
from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from lkybot import config, rag
from lkybot.llm import LLMError

console = Console()
history: list[dict] = []


def _banner() -> None:
    console.print("[green]" + "-" * 60 + "[/green]")
    console.print(
        Panel.fit(
            "[bold]What Would Lee Kuan Yew Do?[/bold]\n"
            "Ask about his life, Singapore's history, geopolitics, or anything else.\n"
            "He answers only from retrieved sources - or says so when the record is silent.",
            border_style="green",
            title="[bold]Lee Kuan Yew chatbot[/bold]",
            title_align="left",
        )
    )
    console.print(
        f"[dim]model[/dim] [cyan]{config.LLM_MODEL}[/cyan] "
        f"[dim]- threshold[/dim] [cyan]{config.SCORE_THRESHOLD:g}[/cyan] "
        f"[dim]- commands:[/dim] /clear /quit"
    )
    console.print("[green]" + "-" * 60 + "[/green]")


def show(result: dict) -> None:
    if result.get("refused"):
        console.print(
            Panel(
                Markdown(result["answer"]),
                title="Lee Kuan Yew",
                border_style="yellow",
            )
        )
        return

    console.print(Panel(Markdown(result["answer"]), title="Lee Kuan Yew", border_style="green"))

    if result.get("flagged_quotes"):
        console.print(
            f"[yellow]Removed {len(result['flagged_quotes'])} quote(s) not found in sources.[/yellow]"
        )

    if result.get("sources"):
        table = Table(title="Sources", show_lines=False, border_style="dim")
        table.add_column("#", justify="right", style="dim")
        table.add_column("Title", style="cyan")
        table.add_column("Section")
        table.add_column("Type")
        table.add_column("Score", justify="right")
        for index, source in enumerate(result["sources"], start=1):
            table.add_row(
                str(index),
                source["title"],
                source["section"],
                source.get("doc_type", "secondary"),
                f"{source['similarity']:.3f}",
            )
        console.print(table)


def main() -> None:
    _banner()
    while True:
        try:
            question = console.input("[bold cyan]You >[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye.")
            break
        if not question:
            continue
        if question in {"/quit", "/exit"}:
            console.print("Goodbye.")
            break
        if question == "/clear":
            history.clear()
            console.print("[dim]History cleared.[/dim]")
            continue
        try:
            with console.status("[green]Thinking...[/green]", spinner="dots"):
                result = rag.answer(question, history)
        except LLMError as exc:
            console.print(f"[red]{exc}[/red]")
            continue
        except Exception as exc:  # noqa: BLE001 - keep the chat loop alive
            console.print(f"[red]Error: {exc}[/red]")
            continue
        show(result)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result["answer"]})


if __name__ == "__main__":
    main()
