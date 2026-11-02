"""Program-unit splitting and the unit-level views built on it."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import ParsedModule
from .source import EXECUTABLE_KINDS, HEADER_KINDS, Statement

_STARTS_IMPLICIT_MAIN = EXECUTABLE_KINDS | {
    "declaration",
    "implicit",
    "dimension",
    "common",
    "equivalence",
    "data",
    "parameter",
    "external",
    "intrinsic",
    "save",
    "format",
    "assign",
    "namelist",
}


@dataclass(slots=True)
class UnitBlock:
    """The statements of one program unit (header and END included)."""

    kind: str  # "subroutine" | "function" | "program" | "block_data"
    name: str
    header: Statement | None
    statements: list[Statement] = field(default_factory=list)
    end: Statement | None = None

    @property
    def start_line(self) -> int:
        return self.statements[0].line if self.statements else 1

    @property
    def end_line(self) -> int:
        return self.statements[-1].end_line if self.statements else 1

    @property
    def body(self) -> list[Statement]:
        """Statements between the header and the END."""
        skip = {id(self.header), id(self.end)}
        return [s for s in self.statements if id(s) not in skip]


def _default_name(kind: str) -> str:
    return "blockdata" if kind == "block_data" else "main"


def split_units(statements: list[Statement]) -> tuple[list[UnitBlock], list[Statement]]:
    """Group logical statements into program units (an unnamed main program is supported).

    Returns the units and the *orphans*: statements outside any unit that cannot start a main
    program (INCLUDE lines, Fortran 90 module wrappers, unparseable text).
    """
    units: list[UnitBlock] = []
    orphans: list[Statement] = []
    current: UnitBlock | None = None
    in_module = in_type = False  # Fortran 90 wrappers whose declarations belong to no unit
    for stmt in statements:
        kind = stmt.kind
        if kind in HEADER_KINDS:
            header_name = getattr(stmt.info, "name", "") or _default_name(kind)
            current = UnitBlock(kind, header_name, stmt, [stmt])
            units.append(current)
        elif kind == "contains":
            orphans.append(stmt)
            current = None
        elif kind == "end":
            if current is not None:
                current.statements.append(stmt)
                current.end = stmt
                current = None
            else:
                in_module = False
        elif current is not None:
            current.statements.append(stmt)
        else:
            compact = stmt.text.upper().replace(" ", "")
            if kind == "f90":
                in_module |= compact.startswith("MODULE") and not compact.startswith("MODULEP")
                if compact.startswith("ENDTYPE"):
                    in_type = False
                elif compact.startswith("TYPE") and not compact.startswith("TYPE("):
                    in_type = True
            if in_module or in_type or kind not in _STARTS_IMPLICIT_MAIN:
                orphans.append(stmt)
            else:
                current = UnitBlock("program", "main", None, [stmt])
                units.append(current)
    return units, orphans


def unit_dependencies(module: ParsedModule) -> dict[str, set[str]]:
    """For each unit, the *other* units of the same module it calls or passes as arguments.

    A reference to an ENTRY point counts as a reference to the unit that owns the entry.
    """
    owner = {u.name: u.name for u in module.units}
    entry_names = {s.name for s in module.api} - owner.keys()
    for unit in module.units:
        owner.update({name: unit.name for name in unit.defines if name in entry_names})
    return {u.name: {owner[r] for r in u.references if r in owner} - {u.name} for u in module.units}
