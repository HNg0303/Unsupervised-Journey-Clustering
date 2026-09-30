# syntax=docker/dockerfile:1

FROM python:3.14-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Only the Python package and its thin development wrappers enter the image.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY scripts ./scripts

# Build the application and all runtime pipeline dependencies as wheels so the
# final stage does not need build tooling or network access.
RUN python -m pip wheel --wheel-dir /wheels \
        ".[pipeline]" \
        "scikit-learn==1.9.0"


FROM python:3.14-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN groupadd --system journey \
    && useradd --system --gid journey --create-home journey \
    && mkdir -p /workspace/data /workspace/models /workspace/outputs \
    && chown -R journey:journey /workspace

COPY --from=builder /wheels /wheels
RUN python -m pip install --no-index --find-links=/wheels \
        "hifpt-journey-clustering[pipeline]" \
    && rm -rf /wheels

WORKDIR /workspace
USER journey

# Override this command with journey-partition, journey-infer, or journey-name
# when running the corresponding batch job.
CMD ["journey-train", "--help"]
