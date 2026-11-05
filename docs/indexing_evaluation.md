# Embeddings, indexes, and evaluation

This contribution adds standalone modules and scripts. Existing parsing, chunking,
BM25, retrieval, benchmark, and serving implementations are unchanged. It uses
those components through their existing interfaces. It does not implement learned
fusion, adaptive cutoff, or LLM serving, which remain with their respective owners.

## Setup

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

Python 3.9+ is supported by this contribution. The model and QASPER data download
on first use. Set `HF_HOME=artifacts/huggingface` to keep downloads in this project.
The comparison CLI runs encoding in a separate worker process to avoid a macOS
OpenMP conflict between PyTorch and FAISS wheels.
Do not set `KMP_DUPLICATE_LIB_OK` to suppress runtime problems.

## Run the experiments

```bash
HF_HOME=artifacts/huggingface .venv/bin/python scripts/run_index_comparison.py --limit 1000 --seed 0 --output results/indexing/qasper.json
HF_HOME=artifacts/huggingface .venv/bin/python scripts/run_index_comparison.py --limit 1000 --seed 0 --scope corpus --output results/indexing/qasper_corpus.json
```

By default both plain and contextual variants run, each with the existing BM25,
exact FAISS `IndexFlatIP`, and approximate `IndexHNSWFlat`. All share chunk
boundaries, questions, and cutoffs. QASPER paper scope searches only the known
paper; corpus scope searches the entire loaded split, not just sampled papers.
The evaluation sample is selected without replacement using a fixed seed from
sorted eligible question IDs. Unanswerable or evidence-less QASPER questions are
excluded by the existing retrieval harness; they remain eligible for answer F1.

Use `--json data/raw/qasper-dev-v0.3.json` for a local split. `--split` applies to
Hugging Face loading; a local file is used as supplied. Use dev data for tuning;
reserve test data for the final run. `--ks 1 3 5 10 20`, `--m 32`,
`--ef-construction 200`, and `--ef-search 64` are configurable. Repeat with
separate output names to study HNSW's quality/latency tradeoff. `--threads` controls
CPU thread counts. `--revision` pins the encoder revision; the existing team's
HFTokenizer uses the model's default tokenizer revision, so changing to a revision
with a different vocabulary requires coordination with the chunking owner.

The script writes:

- JSON: configuration, source-data hash, software/hardware information, metrics,
  breakdowns, individual question rankings, and timing samples.
- CSV: comparison table ready for report analysis.
- `<output_stem>_artifacts/`: chunk provenance, sampled question IDs, indexed text,
  embeddings, sparse postings, and reloadable dense indexes for each variant.
- `artifacts/embeddings/`: caches keyed by model configuration and ordered input
  text. Changing texts, order, or configuration creates a new cache entry. When
  a mutable upstream model revision changes, clear the cache or pin a commit SHA.

Large artifacts and detailed `results/indexing/*.json` logs are ignored by Git.
Compact CSV and Markdown summaries remain available for committing and review.

## Embedding/index API for the retrieval owner

```python
from rag.embedding import SentenceTransformerEmbedder, cached_embeddings
from rag.indexing import FaissRetriever

embedder = SentenceTransformerEmbedder()
vectors = cached_embeddings(embedder, [c.text for c in chunks], "artifacts/embeddings")
retriever = FaissRetriever(embedder, chunks, vectors, kind="flat", scope="paper")
ranked_chunks = retriever.retrieve_batch(questions, k=10)
retriever.save("artifacts/dense")
loaded = FaissRetriever.load("artifacts/dense", embedder)
```

Use the same encoder/revision at query time. On macOS installations with conflicting
OpenMP wheels, run encoding and FAISS search in separate processes as the comparison
CLI does; the in-process API example assumes compatible runtimes. `FaissIndex.search(query_vectors, k)`
returns `(chunk_id, cosine_score)` pairs for candidate fusion. Vectors are converted
to float32 and normalized for both indexing and querying. Invalid vectors and
ambiguous IDs are rejected. Separate per-paper indexes avoid post-filtering a
corpus top-k, which could lose valid results within a paper.

Contextual text prepends paper and section titles within the model token budget.
When a title is too long the prefix is shortened; evidence text is retained. Body
chunks exceeding the budget raise an error instead of silently truncating. This
variant changes embedding/index inputs, not the chunks returned to generation.
Question inputs exceeding the encoder limit are counted in the report; the
encoder's normal truncation behavior applies to them.

## Metric interpretation

- `hit@k`: at least one relevant chunk retrieved. This is not evidence recall.
- `recall@k`: unique gold paragraph/sentence units covered divided by total gold units.
- `precision@k`: relevant chunks divided by k, including unfilled result slots.
- `mrr`: reciprocal rank of the first relevant chunk, truncated at the largest k.
- `ndcg@k`: binary chunk relevance normalized by all relevant chunks in scope.
- `exact_overlap@k`: HNSW neighbor-set agreement with the exact index using the
  same vectors. It measures approximation, not evidence relevance; ties can lower
  set agreement without lowering cosine scores.

Latency is the median of three warmed sequential query batches divided by question
count. It includes Python mapping and normalization but excludes query encoding,
which is reported separately as embedding-or-cache time. It is amortized batch
latency, not a per-request p95 or concurrent-serving benchmark. Embedding cache
hits are not fresh encoding throughput measurements. Dense serialized size counts
FAISS bytes; sparse size counts JSON postings and statistics. These are disk
representations, not process RSS. Build times exclude embeddings and disk writes.
End-to-end generation latency must be collected with the team's actual LLM server.

## HotpotQA

```bash
HF_HOME=artifacts/huggingface .venv/bin/python scripts/run_index_comparison.py --dataset hotpot --json data/raw/hotpot_dev_distractor_v1.json --limit 1000 --output results/indexing/hotpot.json
```

Supply the official list-of-records JSON (`_id`, `question`, `context`,
`supporting_facts`, `answer`). The adapter preserves sentence IDs and groups only
within each article. `paper` scope means all supplied articles for that question.
This is a distractor-context experiment, not a full-Wikipedia retrieval result.
The adapter rejects missing support facts instead of silently lowering denominators.

## Answer evaluation handoff

Generation belongs to the serving owner. Save predictions as a JSON object mapping
question IDs to answer strings or objects with an `answer` field. The evaluator
also accepts HotpotQA's `{"answer": {"id": "text"}}` envelope.

```bash
.venv/bin/python scripts/evaluate_answers.py --gold data/raw/qasper-dev-v0.3.json --predictions results/predictions.json --output results/answer_scores.json
.venv/bin/python scripts/evaluate_answers.py --dataset hotpot --gold data/raw/hotpot_dev_distractor_v1.json --predictions results/hotpot_predictions.json --output results/hotpot_answer_scores.json
```

QASPER F1 takes the best score across annotations, joins extractive spans with
commas, and handles yes/no and unanswerable answers. HotpotQA uses its special
handling of yes/no/noanswer. Missing predictions score zero; they are never dropped.
For an intentionally sampled run, pass `--question-ids` with the saved JSON array.
By default all gold questions are evaluated, including unanswerable questions.
This tool scores answers, not official evidence F1 or HotpotQA joint answer/support
F1. Retrieval evidence is evaluated separately by the retrieval harness.

Followed reference semantics:
- https://github.com/allenai/qasper-led-baseline/blob/main/scripts/evaluator.py
- https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py
- https://github.com/facebookresearch/faiss/wiki/Faiss-indexes

## Sparse-index audit / team follow-up

```bash
.venv/bin/python scripts/audit_sparse_index.py
```

The read-only audit compares numerical scores against `rank_bm25`. It exposes an
existing difference for repeated query tokens and the existing sparse-only result
policy (unmatched/zero-scored documents can be omitted). The owner should resolve
or document these before claiming full reference equivalence. No BM25 code is
changed by this contribution. Reports use the existing implementation as-is.

## Results table and uncertainty

```bash
.venv/bin/python scripts/summarize_index_results.py results/indexing/qasper.json --metric recall@5 --output results/indexing/summary.md
```

This generates a Markdown table with paired bootstrap confidence intervals for
metric differences against the plain exact baseline. It resamples papers rather
than treating multiple questions from one paper as independent. Intervals describe
this dataset sample; they do not establish general superiority or account for
model/parameter selection on the same split.

## Validation completed in this workspace

- 43 automated tests pass, covering the existing pipeline plus index persistence,
  exact cosine ranking, HNSW recovery, scoping, cache invalidation, HotpotQA evidence
  mapping, and answer F1 conventions.
- A real MiniLM/QASPER dev run evaluated all 923 eligible questions across 281
  papers and 11,492 chunks. See `results/indexing/qasper_summary.md` and its CSV.
- Plain exact Recall@5 was 0.4969; contextual exact was 0.5313. The paper-clustered
  bootstrap interval for the difference was +0.0107 to +0.0577. This is a dev-set
  observation, not a final held-out result.
- Per-paper HNSW matched the exact top-20 neighbor sets in this run, but was slower
  at this small per-paper scale. Do not claim HNSW is universally faster.
- The existing parser/evidence matching left 100 of 2,085 evidence strings
  unmatched. The inherited retrieval evaluation uses only matched units; these
  figures are conditional on that mapping. Four fully unmatched questions were
  excluded, along with 60 unanswerable and 18 evidence-less questions.
- HotpotQA's full CLI and answer scorer were checked with a small synthetic fixture
  and real MiniLM embeddings. A full real HotpotQA experiment has not been run.
- No generated LLM answers were provided, so no real answer F1 or end-to-end
  generation latency results are claimed. The scoring CLI is ready for that handoff.

The corpus-scope experiment also completed on the same 923 questions. See
`results/indexing/qasper_corpus_summary.md`. Searching all papers is substantially
harder: plain exact Recall@5 was 0.1222, and contextual exact was 0.1070. Thus the
contextual improvement in known-paper retrieval does not generalize to this
corpus-scope run. Plain HNSW averaged 0.151 ms/query versus 0.448 ms/query for exact
search, excluding encoding. These observations should be reported separately by
scope rather than pooled into one headline result.
