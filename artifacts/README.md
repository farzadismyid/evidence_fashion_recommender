# Final release artefacts

This directory contains the final clean-run manifests, tables, figures, and compact release
package. The canonical index is [`release/release_manifest.json`](release/release_manifest.json);
the accompanying results summary is in
[`../reports/final_clean_run_report.md`](../reports/final_clean_run_report.md).

Large mutable inputs, downloaded data, and embedding arrays are intentionally excluded from Git
and live under the configured `.runtime/current/` root. Local historical material is ignored.
