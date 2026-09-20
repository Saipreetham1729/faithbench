# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] - Unreleased

Initial extraction from the ContextGuard research project.

### Added
- `FaithfulnessChecker` with pluggable methods via a simple `score_claims(claims, evidence)` protocol
- Built-in methods: `rouge_l`, `nli` (both zero-LLM), `llm_judge`, `ragas_faithfulness`,
  `self_consistency` (all three take a user-supplied LLM callable -- provider-agnostic)
- Sentence-level claim extraction
- `CheckResult` / `ClaimResult` with `.overall_score`, `.aggregate_score`, `.status()`, `.to_dict()`
- Optional extras: `[rouge]`, `[nli]`, `[all]`, `[dev]`
- 21 tests covering the checker, claim extraction, and ROUGE-L
