# Generated Outputs

Workflow outputs should be written under `outputs/` or another local directory
configured in `config/paths.yaml`.

Expected output groups:

- `outputs/intermediate/grace/`: filled GRACE storage anomaly and uncertainty tables.
- `outputs/intermediate/pmb/`: monthly PMB component and total tables.
- `outputs/tables/`: regional/basin comparison tables and correction factors.
- `outputs/figures/`: manuscript and supplement figures.

Generated outputs are intentionally ignored by Git. Commit only documentation,
configuration examples, source code, and lightweight tests.
