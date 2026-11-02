"""Canonical registry of legacy (Fortran 77) constructs the tool knows about.

The ids here are the shared vocabulary between the tree-sitter detectors
(``modernizer.parsing``), the retrieval corpus (``modernizer.retrieval``) and the routing
policy (``modernizer.routing``). Add new constructs here first.

``severity``:
  * ``mechanical`` - a local, rule-like rewrite (``DOUBLE PRECISION`` -> ``float``).
  * ``semantic``   - behaviour can change silently or the fix needs wider context
                     (COMMON, EQUIVALENCE, computed GOTO, column-major arrays ...).
``weight`` is the contribution to the module difficulty score.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    id: str
    description: str
    severity: str  # "mechanical" | "semantic"
    weight: float


_SPECS: tuple[FeatureSpec, ...] = (
    # --- control flow ---------------------------------------------------------------------
    FeatureSpec("goto_unconditional", "unconditional GO TO label", "semantic", 2.0),
    FeatureSpec("goto_computed", "computed GO TO (l1, l2, ...), i", "semantic", 3.0),
    FeatureSpec("goto_assigned", "ASSIGN label TO v / assigned GO TO", "semantic", 3.5),
    FeatureSpec("if_arithmetic", "arithmetic IF (e) l1, l2, l3", "semantic", 3.0),
    FeatureSpec("if_logical_single", "single-statement logical IF", "mechanical", 0.5),
    FeatureSpec("do_labeled", "DO loop closed by a statement label", "mechanical", 1.0),
    FeatureSpec(
        "do_shared_terminal", "several DO loops sharing one terminal label", "semantic", 3.0
    ),
    FeatureSpec("do_real_index", "DO loop with a REAL or DOUBLE PRECISION index", "semantic", 2.0),
    FeatureSpec("stop_pause", "STOP or PAUSE statement", "mechanical", 0.5),
    FeatureSpec("alternate_return", "alternate RETURN n and * dummy arguments", "semantic", 3.5),
    FeatureSpec("entry_statement", "ENTRY statement (multiple entry points)", "semantic", 3.5),
    # --- storage and declarations ---------------------------------------------------------
    FeatureSpec("common_block", "COMMON block (shared global storage)", "semantic", 3.0),
    FeatureSpec("equivalence", "EQUIVALENCE (aliased storage)", "semantic", 3.5),
    FeatureSpec("block_data", "BLOCK DATA program unit", "semantic", 2.5),
    FeatureSpec("data_statement", "DATA initialisation", "mechanical", 1.0),
    FeatureSpec("save_statement", "SAVE or implicitly retained local state", "semantic", 1.5),
    FeatureSpec(
        "implicit_typing", "variables typed by the I-N rule (no declaration)", "semantic", 1.5
    ),
    FeatureSpec("implicit_statement", "IMPLICIT typing statement", "mechanical", 1.5),
    FeatureSpec("parameter_statement", "PARAMETER named constants", "mechanical", 0.5),
    FeatureSpec("character_star", "CHARACTER*n fixed-length strings", "semantic", 1.5),
    FeatureSpec("real_kind_star", "REAL*8 / INTEGER*2 style kind suffixes", "mechanical", 0.5),
    FeatureSpec("double_precision", "DOUBLE PRECISION type and D exponents", "mechanical", 0.5),
    FeatureSpec("complex_type", "COMPLEX arithmetic", "semantic", 2.0),
    FeatureSpec("external_intrinsic_stmt", "EXTERNAL or INTRINSIC declarations", "mechanical", 0.5),
    FeatureSpec("statement_function", "statement function definition", "semantic", 2.0),
    # --- arrays ---------------------------------------------------------------------------
    FeatureSpec(
        "array_multidim", "multi-dimensional array (column-major, 1-based)", "semantic", 2.5
    ),
    FeatureSpec("array_custom_bounds", "array with lower bound other than 1", "semantic", 2.0),
    FeatureSpec(
        "array_adjustable",
        "adjustable-size dummy array, dimensions from arguments",
        "semantic",
        2.5,
    ),
    FeatureSpec("array_assumed_size", "assumed-size dummy array A(*)", "semantic", 2.0),
    FeatureSpec(
        "sequence_association", "array element passed where an array is expected", "semantic", 3.5
    ),
    # --- procedures and arguments ---------------------------------------------------------
    FeatureSpec(
        "arg_output",
        "dummy argument assigned inside the routine (pass by reference)",
        "semantic",
        2.5,
    ),
    FeatureSpec("procedure_argument", "procedure passed as an argument", "semantic", 2.5),
    FeatureSpec(
        "machine_constants", "D1MACH/R1MACH/I1MACH style machine constant calls", "semantic", 2.0
    ),
    # --- input and output -----------------------------------------------------------------
    FeatureSpec("format_statement", "FORMAT statement and edit descriptors", "semantic", 2.5),
    FeatureSpec("io_formatted", "formatted READ/WRITE/PRINT", "semantic", 2.0),
    FeatureSpec("io_list_directed", "list-directed READ/WRITE/PRINT (*)", "mechanical", 1.0),
    FeatureSpec("io_file_control", "OPEN/CLOSE/REWIND/BACKSPACE/INQUIRE", "semantic", 2.0),
    FeatureSpec("hollerith", "Hollerith constant (nHtext)", "mechanical", 1.5),
    # --- expressions and intrinsics -------------------------------------------------------
    FeatureSpec("integer_division", "integer division (truncation toward zero)", "semantic", 2.0),
    FeatureSpec(
        "intrinsic_specific",
        "type-specific intrinsic names (DABS, AMAX1, IFIX ...)",
        "mechanical",
        0.5,
    ),
    FeatureSpec(
        "intrinsic_semantics",
        "SIGN/MOD/NINT/INT/AINT/DIM with Fortran rounding rules",
        "semantic",
        1.5,
    ),
    FeatureSpec("bit_intrinsics", "IAND/IOR/IEOR/ISHFT/NOT bit operations", "mechanical", 1.0),
    FeatureSpec("logical_dotted", "dotted relational and logical operators", "mechanical", 0.5),
    FeatureSpec("character_substring", "substring A(i:j) and // concatenation", "semantic", 1.5),
    FeatureSpec("exponent_literal_d", "D exponent real literal (1.0D0)", "mechanical", 0.5),
    # --- source form and dialect ----------------------------------------------------------
    FeatureSpec(
        "fixed_form_source", "fixed-form columns, labels and continuation lines", "mechanical", 0.5
    ),
    FeatureSpec(
        "vendor_extension",
        "non-standard extension (INCLUDE, STRUCTURE, DO WHILE, NAMELIST ...)",
        "semantic",
        2.0,
    ),
)

FEATURES: dict[str, FeatureSpec] = {s.id: s for s in _SPECS}
FEATURE_IDS: tuple[str, ...] = tuple(FEATURES)


def weight_of(feature_id: str) -> float:
    return FEATURES[feature_id].weight


def is_semantic(feature_id: str) -> bool:
    return FEATURES[feature_id].severity == "semantic"
