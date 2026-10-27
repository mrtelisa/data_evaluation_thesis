"""Report testuale (in inglese, come la tesi), scritto in statistics/report.txt e
stampato a terminale."""

from pathlib import Path


class Report:
    def __init__(self, title: str):
        self.lines = [title, "=" * len(title), ""]

    def section(self, title: str):
        self.lines += ["", title, "-" * len(title)]

    def text(self, line: str = ""):
        self.lines.append(line)

    def save(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        text = "\n".join(self.lines) + "\n"
        (out_dir / "report.txt").write_text(text, encoding="utf-8")
        print(text)
