# Privacy and threat model

## Intended boundary

This starter is intended for one user on one computer. Streamlit and FastAPI bind to loopback. Ollama is called at `127.0.0.1`. Uploaded document contents are not intentionally persisted; extraction uses a temporary file and the model prompt remains in process memory. Legal-source text and embeddings are persisted locally under `local_data/`.

## Out of scope

This is not hardened for hostile multi-user environments, public hosting, compromised operating systems, disk forensics, backups, swap, or a malicious dependency. Local storage is not encrypted by this app. The user should not expose ports or use highly sensitive documents until security controls are reviewed.

## Required verification

1. Download all models/dependencies, then disable networking and run a complete analysis.
2. Monitor connections during source indexing and document analysis.
3. Inspect dependencies for telemetry and remote calls.
4. Inspect temp directories, logs, process environment, and local index files.
5. Confirm no content is sent to external services.
6. Document any network access that remains and make optional/off by default.
