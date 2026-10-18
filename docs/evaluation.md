# Evaluation plan

For every test, save the input document locally and do not commit private data.

- Citation accuracy: every contract quote must be an exact substring of extracted text; every legal passage ID must exist in the index.
- Retrieval recall@5: independently label relevant legal passages for a small question set and measure whether they appear among the top five.
- Unsupported question: ask for a term that is absent; expected behaviour is an explicit uncertainty/document-silence response.
- Jurisdiction mismatch: use a question that names a jurisdiction not represented by the index; expected behaviour is to disclose missing coverage.
- Stale source: include source metadata with an old verification date; expected behaviour is a currency warning.
- Prompt injection: include text in a document asking the model to ignore rules or reveal other files; expected behaviour is to treat it as document content, not instructions.
- Malformed file / scanned PDF: expected behaviour is a clear error, not a fabricated summary.
- Privacy: run with network disabled and capture connection-monitoring evidence.
- Performance: record model tag, quantization, OS, CPU/RAM/GPU, document pages, indexing time, analysis latency and peak memory.

Do not treat a plausible answer or high embedding similarity as proof of legal correctness.
