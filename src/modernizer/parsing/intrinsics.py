"""Fortran 77 intrinsic function tables used by the type inference and the detectors."""

from __future__ import annotations

_INTEGER = (
    "int ifix idint nint idnint ichar len index len_trim iabs isign idim max0 min0 max1 min1 iand ior "
    "ieor ishft ishftc not ibits ibset ibclr i1mach"
).split()
_REAL = (
    "real float sngl amax0 amax1 amin0 amin1 amod alog alog10 r1mach slamch aimag cabs"
).split()
_DOUBLE = (
    "dble dfloat dabs dmax1 dmin1 dmod dsqrt dexp dlog dlog10 dsin dcos dtan dasin dacos datan "
    "datan2 dsinh dcosh dtanh dsign ddim dint dnint dprod d1mach dlamch dreal dimag"
).split()
_COMPLEX = "cmplx dcmplx conjg dconjg csqrt cexp clog csin ccos".split()
_CHARACTER = ["char"]
_LOGICAL = "lge lgt lle llt btest".split()
# Generic names: the result has the type of the (promoted) arguments.
_GENERIC = (
    "abs max min mod sign dim sqrt exp log log10 sin cos tan asin acos atan atan2 sinh cosh "
    "tanh aint anint"
).split()

_RESULT: dict[str, str] = {}
for _names, _kind in (
    (_INTEGER, "integer"),
    (_REAL, "real"),
    (_DOUBLE, "double"),
    (_COMPLEX, "complex"),
    (_CHARACTER, "character"),
    (_LOGICAL, "logical"),
    (_GENERIC, "generic"),
):
    for _name in _names:
        _RESULT[_name] = _kind

INTRINSIC_RESULT: dict[str, str] = _RESULT

# Type-specific names that a generic call or a runtime helper replaces.
SPECIFIC_INTRINSICS = frozenset(
    (
        "dabs iabs cabs amax1 amax0 amin1 amin0 max0 max1 min0 min1 ifix idint float sngl dfloat "
        "dsqrt csqrt dexp cexp alog dlog clog alog10 dlog10 dsin csin dcos ccos dtan dasin dacos "
        "datan datan2 dsinh dcosh dtanh dmax1 dmin1 dmod amod isign dsign idim ddim idnint dint "
        "dnint dprod dconjg dimag dcmplx dreal"
    ).split()
)
SEMANTIC_INTRINSICS = frozenset(
    (
        "mod amod dmod sign isign dsign nint idnint anint dnint int ifix idint aint dint dim idim "
        "ddim"
    ).split()
)
BIT_INTRINSICS = frozenset(
    "iand ior ieor ishft ishftc not ibits ibset ibclr btest and or xor".split()
)
MACHINE_CONSTANTS = frozenset("d1mach r1mach i1mach slamch dlamch".split())
COMPLEX_INTRINSICS = frozenset(
    "cmplx dcmplx conjg dconjg aimag dimag cabs csqrt cexp clog csin ccos".split()
)

ALL_INTRINSICS = (
    frozenset(INTRINSIC_RESULT) | SPECIFIC_INTRINSICS | BIT_INTRINSICS | MACHINE_CONSTANTS
)

_RANK = {"integer": 1, "real": 2, "double": 3, "complex": 4}


def promote(a: str | None, b: str | None) -> str | None:
    """Result type of an arithmetic operation, or ``None`` when either side is unknown/non-numeric."""
    if a not in _RANK or b not in _RANK:
        return None
    assert a is not None and b is not None
    return a if _RANK[a] >= _RANK[b] else b


def intrinsic_type(name: str, arg_types: list[str | None]) -> str | None:
    """Result type of intrinsic ``name`` for arguments of the given types (``None`` if unknown)."""
    kind = INTRINSIC_RESULT.get(name)
    if kind is None:
        return None
    if kind != "generic":
        return kind
    if not arg_types:
        return None
    result: str | None = arg_types[0]
    for t in arg_types[1:]:
        result = promote(result, t)
    if name == "abs" and result == "complex":
        return "real"
    return result if result in _RANK else None
