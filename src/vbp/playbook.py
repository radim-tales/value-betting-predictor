from __future__ import annotations
from dataclasses import dataclass, field

SECTION_ORDER = ["Priors", "Rules", "Hypotheses", "Banned", "Notes"]


@dataclass
class Playbook:
    sections: dict[str, list[str]] = field(default_factory=lambda: {s: [] for s in SECTION_ORDER})

    @classmethod
    def parse(cls, text: str) -> "Playbook":
        sections = {s: [] for s in SECTION_ORDER}
        current = None
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("## "):
                name = stripped[3:].strip()
                current = name if name in sections else None
            elif stripped.startswith("- ") and current:
                sections[current].append(stripped[2:].strip())
        return cls(sections=sections)

    def serialize(self) -> str:
        parts = []
        for s in SECTION_ORDER:
            parts.append(f"## {s}")
            parts.extend(f"- {item}" for item in self.sections.get(s, []))
            parts.append("")
        return "\n".join(parts).rstrip() + "\n"

    def enforce_limits(self, max_chars: int, max_rules: int) -> None:
        """Trim to fit: cap Rules count, then drop Notes/Banned/Hypotheses until under max_chars."""
        self.sections["Rules"] = self.sections["Rules"][:max_rules]
        trim_order = ["Notes", "Banned", "Hypotheses"]
        while len(self.serialize()) > max_chars:
            for s in trim_order:
                if self.sections[s]:
                    self.sections[s].pop()
                    break
            else:
                # nothing left to trim in soft sections; trim rules as last resort
                if self.sections["Rules"]:
                    self.sections["Rules"].pop()
                else:
                    break
