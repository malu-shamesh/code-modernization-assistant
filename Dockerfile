# Cloud Run image for the modernizer HTTP service.
#
#   docker build -t code-modernizer .
#   docker run --rm -p 8080:8080 -e MODERNIZER_PROVIDER=mock code-modernizer
#
# The package is installed (not mounted) so the image has no build tooling or sources at runtime.
# The core dependencies (tree-sitter and tree-sitter-fortran for parsing, numpy and pytest for
# the generated-test sandbox) come with the package; the f77rt runtime that converted code
# imports ships inside it (modernizer/runtime/f77rt.py) and the test runner copies it into every
# sandbox. The slim base has no compiler, so the install relies on prebuilt wheels.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Packaging metadata and sources first; hatchling reads README.md and needs src to build.
WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[gcp,serve]"

# Fail the build, not the first request, when an import the pipeline needs is missing. Runs from
# / so that the installed package is imported and not the sources in /build/src.
WORKDIR /
RUN python -c "import numpy, pytest, tree_sitter, tree_sitter_fortran; \
    from modernizer.runtime import runtime_source; \
    from modernizer.llm import PricingTable; \
    assert 'def idiv' in runtime_source(); \
    assert PricingTable.default().lookup('gemma-3-12b-it') is not None"

# Generated tests are untrusted code and run in this container, so keep the process unprivileged.
# The home directory is writable because the test runner passes HOME on to pytest.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app
USER app
WORKDIR /app

EXPOSE 8080

# Shell form through sh -c so that $PORT (set by Cloud Run) expands; exec keeps uvicorn as PID 1
# so it receives SIGTERM directly and can shut down gracefully.
CMD ["sh", "-c", "exec uvicorn --factory modernizer.service.app:create_app --host 0.0.0.0 --port ${PORT:-8080}"]
