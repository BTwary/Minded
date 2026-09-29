# Hugging Face Space viability

The existing `hf_space/` application is viable as a public, non-authoritative demonstration of the analytical ideas. It is **not** the deployment target for the local-first product.

### Appropriate role

- public product demo
- synthetic benchmark showcase
- small temporary user-file experiments
- visual explanation of DuckDB/SciPy/Bayesian concepts

### Explicit limits

- 50 MB upload limit in the demo
- in-memory/temporary processing
- no durable local AA-OS workspace
- no user-controlled encrypted backup
- no production authentication/tenancy contract
- benchmark data is synthetic and labelled as such

### Recommendation

Keep the Space as a lightweight showcase. Do not make the production local application depend on Hugging Face.
