# Pull-request validation and review record

Date: 2026-10-09

## Pull request

- URL: `https://github.com/SpikeSpiegelJ/Detect/pull/2`
- Head branch: `codex/sci-evidence-implementation`
- Base branch: `codex/initial-research-import`
- Rationale for the stacked base: pull request 1 still owns the initial repository import; pull request 2 isolates the subsequent experiment, evidence, and manuscript work.

## Validation

- All 33 research-specific tests passed across split integrity, annotation provenance, reviewed-label merging, geometry profiling, specialist data construction, prediction fusion, feeder relations, weak-error review, and CSL losses.
- Every Python file changed relative to the PR base passed `ruff format --check`.
- Every changed Python file passed Ruff import-order and critical correctness checks (`E9`, `F63`, `F7`, and `F82`).
- `git diff --check` reported no whitespace errors.
- The live GitHub head before this documentation-only record was `6025b90326398764f3676874cc997b691eb1e8b8`.
- GitHub reported zero check runs and zero workflow runs for the PR head. Therefore, this fork has no automated format or review result to pull or address.

## Adversarial review gates

1. **What could have been deleted instead of added?** The earlier elongated-target regression-weighting implementation, configuration, presets, and dedicated tests were deleted after its mechanism was contradicted. The later CSL modules are retained only because their exact implementations and negative controlled results are part of the reproducibility record; all are disabled by default and excluded from the final performance claim.
2. **Does an added condition suppress a symptom rather than relocate a trigger?** No. Gain checks enable explicit experimental loss terms, and attention capture is owned by the context module and consumed by the detection loss. They do not hide an error or suppress a warning. Default-zero configurations preserve the original detector path.

## Full-diff conclusion

No blocking correctness, data-leakage, or unsupported-claim finding remained after formatting normalization and the 33-test validation. The rejected interventions remain labelled as rejected throughout the evidence package and manuscript framing. The PR is ready for human review but must not be merged automatically while the stacked base PR remains open.
