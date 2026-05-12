# Hybrid Retrieval with RRF

This folder runs hybrid retrieval in three stages:

1. Parallel Retrieval: run BM25 and the finetuned bi-encoder in parallel.
2. Merging: use RRF to fuse the two ranked lists into the top N candidates.
3. Reranking: pass the top N candidates through the cross-encoder to produce the final ranking.

The two initial retrieval sources are:

- BM25 from `traditional-retrieval/source/utils/BM25_retriever.py`
- Finetuned dense bi-encoder from `pretrain-retrieval/src/retrieval_pipeline/retrieval/dense_retriever.py`

RRF does not require score normalization between BM25 and the neural model. It only uses ranks:

```text
RRF(doc) = sum(1 / (rrf_k + rank_i(doc)))
```

With `rrf_k=60`, a document ranked first by both BM25 and the finetuned model receives:

```text
1 / (60 + 1) + 1 / (60 + 1)
```

## Try it

From the project root:

```powershell
python hybrid-retrieval/search_rrf.py --query "shock wave boundary layer interaction"
```

Print a short document content preview:

```powershell
python hybrid-retrieval/search_rrf.py --query "shock wave boundary layer interaction" --show-text
```

Change the number of candidates before fusion:

```powershell
python hybrid-retrieval/search_rrf.py `
  --query "shock wave boundary layer interaction" `
  --bm25-k 100 `
  --finetuned-k 100 `
  --rrf-top-n 50 `
  --final-k 10 `
  --rrf-k 60
```

## Use a different config

By default, the script uses the finetuned config:

```text
pretrain-retrieval/configs/finetuned.json
```

To try the pretrained MiniLM config:

```powershell
python hybrid-retrieval/search_rrf.py `
  --finetuned-config pretrain-retrieval/configs/pretrain.json `
  --query "shock wave boundary layer interaction"
```

## Evaluate

```powershell
python hybrid-retrieval/evaluate_rrf.py 
```
