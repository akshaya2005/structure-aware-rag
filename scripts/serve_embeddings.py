"""CLI: serve all-MiniLM-L6-v2 embeddings over HTTP.

Usage: python3 scripts/serve_embeddings.py [--host 127.0.0.1] [--port 8000] [--device cpu]

    curl -s localhost:8000/embed -H 'content-type: application/json' \
        -d '{"texts": ["What datasets are used?"]}'
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import uvicorn  # noqa: E402

from rag.embedding import DEFAULT_MODEL_NAME, EmbedderConfig, SentenceTransformerEmbedder  # noqa: E402
from rag.serving import create_app  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default=None, help="cpu | cuda | mps (default: auto)")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    embedder = SentenceTransformerEmbedder(
        EmbedderConfig(model_name=args.model, device=args.device, batch_size=args.batch_size)
    )
    uvicorn.run(create_app(embedder), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
