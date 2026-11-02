"""The Fortran 77 to Python calling convention shared by the converter, the validator and the
test generator.

Everything that has to agree between "what the converted code looks like" and "how tests call
it" lives in this one text, so prompts cannot drift apart. ``modernizer.parsing`` derives the
per-unit contract (parameter names, kinds, which scalars are written) from the Fortran AST;
this text explains how that contract maps onto Python.
"""

from __future__ import annotations

RUNTIME_MODULE = "f77rt"

CALLING_CONVENTION = f"""\
FORTRAN 77 -> PYTHON CALLING CONVENTION (follow exactly; the result is machine-validated and tested)

Program units
- Every SUBROUTINE and FUNCTION becomes one module-level Python function. Its name is the
  Fortran name in lower case. EVERY named dummy argument stays a Python parameter, in the
  Fortran order and with the lower-case Fortran name, including pure outputs (callers pass a
  placeholder that the function ignores and overwrites). Do not add, drop or reorder
  parameters. '*' alternate-return dummies are the only dummies that are not parameters. A
  dummy whose name is a Python keyword gets a trailing underscore (IN -> in_).
- Every ENTRY statement becomes its own module-level function with its own dummy arguments,
  sharing the module-level state of the unit that contains it.
- A PROGRAM becomes def main(): ... plus an 'if __name__ == "__main__": main()' guard.
- BLOCK DATA needs no callable: it initialises the module-level state of the COMMON blocks it
  names when the module is imported. ALL initial state (BLOCK DATA, DATA, COMMON initial
  values) is built at import time, so re-importing or reloading the module resets it.

Arguments and results (Fortran passes by reference, Python does not)
- Scalar dummy arguments are passed by value. A scalar dummy argument that the routine assigns
  (listed in the unit contract as OUTPUTS) is RETURNED, in parameter order.
- FUNCTION: return the function result first, then the outputs, as a tuple
  (result, out1, out2, ...); with no outputs return the bare result.
- SUBROUTINE: no outputs -> return None; one output -> return that value; several -> a tuple.
- Alternate returns (RETURN n with * dummy arguments): the returned tuple ends with an int,
  the number of the alternate return taken (0 for a normal RETURN). A subroutine whose only
  result is that int returns it bare.
- Array dummy arguments are numpy.ndarray objects allocated by the caller and modified IN
  PLACE (they are never returned). Fortran subscript i maps to Python index i-1; a
  multi-dimensional A(i, j) is a.shape == the declared extents indexed as a[i-1, j-1]. An
  assumed-size or adjustable array takes its extents from the array object, but the dimension
  arguments (N, LDA ...) remain parameters and keep controlling the loops. A lower bound other
  than 1, A(0:n), shifts the index by that bound. CHARACTER arrays are Python lists of str.
- EXTERNAL dummy arguments (procedures) are Python callables following the same convention.

Types and arithmetic
- INTEGER -> int. Integer division truncates toward zero (use the runtime idiv, never // on
  possibly negative operands, never / between integers). REAL and DOUBLE PRECISION -> float
  (IEEE double). COMPLEX -> complex. LOGICAL -> bool. CHARACTER*n -> str, space padded or
  truncated to n on assignment; comparisons ignore trailing blanks.
- Use the runtime helpers for Fortran-specific intrinsics: SIGN, MOD (sign of the dividend),
  NINT (half away from zero), INT and AINT (truncate), DIM, and the type-specific names
  (DABS, AMAX1, IFIX ...), exponentiation with an integer exponent, and so on.
- An array element passed where the callee expects an array (sequence association) becomes a
  numpy view of the remaining elements (a[i-1:] for 1-D, a.reshape(-1, order='F')[k:] for
  a column-major tail) so the callee's in-place writes reach the caller's array.
- A DO loop (use do_range / do_trips) evaluates start, limit and step once; the trip count is
  max(0, (limit - start + step) // step) computed with truncating division; the loop variable
  keeps start + trips * step afterwards. Never modify the loop variable inside the body.

Control flow
- Remove GOTO, computed GOTO, arithmetic IF and shared DO terminal labels by restructuring
  into if/elif/else, while, for, break/continue when the flow is structured. When it is not
  (jumps into loops, backward jumps with several entry points), emit a state machine: a
  'label' variable and a 'while True:' loop with an if/elif chain per basic block.
- STOP -> raise f77rt.FortranStop(code) inside subroutines and functions; a STOP that is the
  final statement of a PROGRAM simply ends main() (return). PAUSE -> no-op.

State and I/O
- COMMON blocks become module-level state shared by every function of the module (for
  example a small class instance per block, with the Fortran variable names as attributes).
  EQUIVALENCE and SAVE/DATA-initialised locals that persist between calls need explicit
  module-level state.
- WRITE/PRINT to the standard unit (* or 6) print to sys.stdout using the runtime FORMAT
  helpers, preserving the column layout (no carriage-control interpretation, like gfortran).
  READ from the standard input unit reads lines from sys.stdin. Other units are files opened
  through the runtime helpers.

Module shape
- One self-contained .py file. Imports: the standard library, numpy and the runtime module
  `{RUNTIME_MODULE}` (shipped next to the converted file; imported as: from {RUNTIME_MODULE} import ...).
- Keep the Fortran comments as Python comments and keep the order of the program units.
"""
