# Installation

PtyLabX requires **Python 3.10+** and uses [uv](https://docs.astral.sh/uv/) for dependency management.

## From source (pip)

```bash
pip install git+https://github.com/ShantanuKodgirwar/PtyLabX.git
```

### With GPU support

=== "CUDA 13 (Recommended)"

    ```bash
    pip install "ptylabx[cuda13]@git+https://github.com/ShantanuKodgirwar/PtyLabX.git"
    ```

=== "CUDA 12"

    ```bash
    pip install "ptylabx[cuda12]@git+https://github.com/ShantanuKodgirwar/PtyLabX.git"
    ```

## Development setup

```bash
git clone https://github.com/ShantanuKodgirwar/PtyLabX.git
cd PtyLabX
uv sync
```

This creates a virtual environment at `.venv/`. Activate it in your IDE or terminal:

```bash
source .venv/bin/activate
```

### With GPU and dev dependencies

```bash
uv sync --extra cuda13 --group dev
```

### Verify JAX backend

```bash
python -c "import jax; print(jax.default_backend())"
```

This should print `gpu` for GPU installations or `cpu` for CPU-only.

## Running tests

```bash
uv run python -m pytest tests/ -v -s
```

## Serving documentation

```bash
uv sync --group docs
uv run mkdocs serve
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.
