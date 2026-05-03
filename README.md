# Retrieval Pipeline

- `sentence-transformers/all-MiniLM-L6-v2` to create embedding
- FAISS CPU use for indexing and dense retrieval
- `cross-encoder/ms-marco-MiniLM-L6-v2` to rerank the results
- Dataset in `cranfield-dataset/`

## Overall pipeline

```mermaid
flowchart TD
    A["Input query"] --> B["Load config Settings"]
    B --> C["RetrievalPipeline.from_settings"]

    C --> D["Load EmbeddingModel"]
    D --> D1["SentenceTransformer<br/>indexing.embedder_model"]

    C --> E["Load FAISS index"]
    E --> E1["index.faiss"]
    E --> E2["documents.json"]

    C --> F{"reranking.enabled ?"}
    F -->|true| G["Load RerankerModel"]
    G --> G1["CrossEncoder<br/>reranking.model"]
    F -->|false| H["Do not load reranker"]

    D1 --> I["DenseRetriever"]
    E1 --> I
    E2 --> I
    G1 --> J["RetrievalPipeline ready"]
    H --> J
    I --> J

    J --> K["search(query, top_k)"]
    K --> L["requested_top_k = top_k<br/>or reranking.top_k_final"]
    L --> M["candidate_k = max(reranking.top_k_candidates, requested_top_k)"]
    M --> N["DenseRetriever.search(query, candidate_k)"]
    N --> O["EmbeddingModel.encode_query(query)"]
    O --> P["Query vector float32"]
    P --> Q["FaissIndex.search(query_vector, candidate_k)"]
    Q --> R["Candidate SearchHit list<br/>doc_id, FAISS score, text"]

    R --> S{"Have reranker ?"}
    S -->|No| T["final_top_k = top_k<br/>or indexing.top_k"]
    T --> U["Return hits[:final_top_k]"]

    S -->|Yes| V["Create pairs [query, hit.text]"]
    V --> W["CrossEncoder predict score"]
    W --> X["Sort in decending order reranker score"]
    X --> Y["Return top requested_top_k"]
```

## Run

### 1. Build FAISS index

```powershell
$env:PYTHONPATH="src"
python -m retrieval_pipeline.cli.build_index --config configs/base.json
```

After the process is complete, index folder will have:

- `index.faiss`
- `documents.json`

### 2. Test query

```powershell
$env:PYTHONPATH="src"
python -m retrieval_pipeline.cli.search --config configs/base.json --query "shock wave boundary layer interaction"
```

Result format:

```text
rank    doc_id    score
```

### 3. Evaluate on Cranfield

Metric `precision@k`, `recall@k`, `map@k`, `ndcg@k`
with the default top-k = `10` (can adjust with `--top-k` flag)

```powershell
$env:PYTHONPATH="src"
python -m retrieval_pipeline.cli.evaluate --config configs/base.json
```

```powershell
$env:PYTHONPATH="src"
python -m retrieval_pipeline.cli.evaluate --config configs/base.json --top-k 10
```