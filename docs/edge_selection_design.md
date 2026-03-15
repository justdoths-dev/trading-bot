# Edge Selection Engine Design

## 1. Purpose and Non-Goals

### Purpose
The Edge Selection Engine is a future research-to-decision bridge that will evaluate research outputs and produce a conservative, explainable selection candidate in shadow mode before any live activation is considered.

Its design objective is to answer a narrow question:

Given the current research snapshot, which symbol-strategy-horizon combination, if any, would be the most defensible candidate for observation under strict safety gates?

The engine is designed to:
- consume existing research-layer outputs only
- combine recent-window and cumulative evidence
- prioritize stability, sample sufficiency, and drift awareness over aggressiveness
- support explicit abstention when evidence is weak, unstable, contradictory, or operationally unsafe
- emit structured outputs suitable for schema validation, testing, and later audit review
- remain decoupled from execution and capital deployment until much later activation phases

### Non-goals
This phase does not include:
- live order generation
- execution integration
- capital allocation
- position sizing
- stop loss or take profit logic
- broker or exchange connectivity
- AI-only decision authority
- automatic activation of a selected candidate
- forced ranking when abstention is safer

AI remains an interpreter and summarization layer, not the final decision maker.

## 2. Why Live Activation Is Not Allowed Yet

Live activation is explicitly blocked at this stage for the following reasons:
- The retained dataset is approximately 3578 records, which is meaningful for research but still limited for production-grade edge activation across symbols, strategies, horizons, and changing market regimes.
- The latest summary window is approximately 124 records, which is too small to justify live selection confidence in a regime-sensitive environment.
- Research outputs already show that drift and instability can emerge quickly, and those effects must be observed over time before any activation is considered.
- Current infrastructure is strong enough for shadow-mode observation, but not yet for live confidence calibration.
- Selection quality, abstain behavior, ranking persistence, and false-positive rates still need testing, back-checking, and operational review.
- The system does not yet include mature position lifecycle management or capital allocation controls, so selection authority must remain observational only.

Therefore, the engine is design-only now and should move to shadow mode before any limited activation is considered.

## 3. Required Input Signals for Future Selection

The future engine should consume research-layer outputs only. The required inputs are:

1. Latest research summary  
Source: `logs/research_reports/latest/summary.json`  
Primary use:
- recent-window dataset size
- strategy snapshot
- symbol distribution
- horizon-level highlights
- latest-window context

2. Comparison report  
Source: `logs/research_reports/comparison/summary.json`  
Primary use:
- recent versus cumulative comparison context
- candidate agreement or disagreement
- comparative edge visibility

3. Edge stability scores  
Source: `logs/research_reports/edge_scores/summary.json`  
Primary use:
- category-level stability scoring
- source preference selection
- visible horizon context
- stability labels for strategy, symbol, and alignment state

4. Edge score history  
Source: `logs/research_reports/edge_scores_history.jsonl`  
Primary use:
- longitudinal history of scored groups
- persistence of edge observations over time
- trend monitoring for later calibration work
- consecutive-cycle visibility for future stability duration checks

5. Score drift report  
Source: `logs/research_reports/score_drift/summary.json`  
Primary use:
- recent score changes
- stability transitions
- horizon-count changes
- drift visibility for ranking penalties or hard blocks

## 4. Optional / Derived Fields

The following fields are not mandatory as raw inputs but should be derived if available:
- latest_window_record_count
- cumulative_record_count
- latest_candidate_strength
- cumulative_candidate_strength
- selected_candidate_strength
- selected_stability_label
- selected_visible_horizons
- drift_direction
- score_delta magnitude
- stability_transition
- horizon_count_delta
- source_preference consistency
- symbol-level support count
- strategy-level support count
- alignment-state agreement indicator
- composite qualification score
- selection_confidence
- abstain reason set
- consecutive_visible_cycles
- consecutive_stable_cycles
- symbol_cumulative_support
- strategy_cumulative_support

### Derived field interpretation rules
- `selected_candidate_strength` must be derived from upstream research scoring outputs and must not be independently recomputed by the selection engine using undocumented logic.
- `selection_confidence` is a normalized, shadow-only interpretive field intended for calibration and review. It must not be treated as execution authority.
- `consecutive_visible_cycles` and `consecutive_stable_cycles` should be derived from edge score history and must remain explainable from retained upstream records.

These derived fields should remain explainable and traceable to upstream research outputs.

## 5. Edge Qualification Criteria

A future candidate should only qualify for ranking if all baseline conditions are satisfied:
- the candidate is observable from existing research outputs
- symbol, strategy, and horizon are all identifiable
- latest-window evidence is present
- cumulative evidence is present
- latest and cumulative evidence are not structurally contradictory without explanation
- candidate strength is not `insufficient_data`
- stability evidence is visible at the selected source-preference layer
- no hard drift block is active
- no minimum-sample block is active
- no symbol-history block is active
- required upstream reports are present, parseable, and not stale

A candidate may still be ranked conservatively if evidence is mixed but not blocked. However, ranking eligibility is separate from final selection eligibility.

## 6. Minimum Sample Thresholds

The engine should be sample-aware and conservative. A future implementation should enforce both hard minimums and preferred minimums.

### Hard minimum thresholds
A candidate must be blocked if any of the following is true:
- latest-window candidate sample support is below 20
- cumulative candidate sample support is below 60
- latest-window visible horizon count is 0 for the selected source
- cumulative history for the candidate category is missing when cumulative evidence is required
- symbol cumulative support is below 150
- strategy cumulative support is below 120

### Preferred thresholds
A candidate should be penalized if any of the following is true:
- latest-window support is between 20 and 39
- cumulative support is between 60 and 119
- symbol cumulative support is between 150 and 249
- strategy cumulative support is between 120 and 199
- the candidate is visible in only one horizon
- the candidate appears only in a narrow symbol or strategy slice without broader support

These numbers are intentionally conservative and should be revised only after shadow-mode evidence is collected.

## 7. Stability Score Gate

The future engine should not treat raw score magnitude alone as enough.

### Hard gate
Block selection if:
- selected stability label is `insufficient_data`
- selected stability label is `unstable`
- selected edge stability score is below a minimum activation score threshold
- consecutive visible cycles are below the minimum duration requirement
- consecutive stable cycles are below the minimum duration requirement

### Recommended initial score threshold
For future limited activation review, require:
- edge stability score >= 3.0 for shadow-mode candidate visibility
- edge stability score >= 4.0 for limited-activation consideration

### Recommended initial duration threshold
Require:
- consecutive visible cycles >= 2 for shadow-mode candidate visibility
- consecutive stable cycles >= 3 for limited-activation consideration

### Preferred stability states
The strongest candidates should have:
- `single_horizon_only` only for observation, not activation
- `multi_horizon_confirmed` for any serious escalation review

The engine should prefer persistence over short-lived strength spikes.

## 8. Drift Penalty / Drift Block Policy

Drift must be visible and actionable in the output.

### Hard drift block
Block selection if any of the following is true for the candidate:
- drift direction is `decrease` and score delta is materially negative
- stability transition moved from a stronger state to a weaker state
- horizon_count_delta is negative in a way that reduces confirmation breadth
- repeated recent drift events indicate deterioration rather than noise
- source preference changed recently while the candidate weakened rather than strengthened

### Soft drift penalty
Penalize ranking if:
- drift direction is `flat` after a prior stronger state
- source preference changed recently without stronger confirmation
- horizon breadth is unchanged but candidate strength weakened
- score deterioration exists but is not severe enough to hard-block

### Positive drift handling
Positive drift may improve ranking, but must not override:
- sample insufficiency
- symbol history insufficiency
- instability
- malformed or stale upstream inputs

## 9. Abstain / No-Selection Policy

The engine must be allowed to abstain.

Abstention is the default-safe outcome when confidence is not sufficient.

### Abstain conditions
Abstain if:
- no candidate clears minimum sample thresholds
- no candidate clears symbol-history thresholds
- no candidate clears stability gates
- drift blocks all viable candidates
- latest-window and cumulative evidence materially disagree
- ranking differences are too small to justify preference
- all candidates are effectively weak or unstable
- upstream research files are missing, stale, or malformed
- the leading candidate lacks sufficient persistence across recent cycles
- candidate confidence cannot be justified in an explainable way

### Abstain output requirement
The output must include explicit machine-readable reason codes describing why no selection was made.

Abstention is preferable to a weak, unstable, or operationally ambiguous selection.

## 10. Ranking and Tie-Break Rules

Ranking should be deterministic and conservative.

### Primary ranking order
1. hard-gate pass status
2. higher stability gate score
3. higher stability duration quality
4. larger cumulative support
5. larger symbol cumulative support
6. larger recent-window support
7. stronger selected candidate strength
8. broader visible horizon coverage
9. lower drift penalty
10. higher consistency between recent and cumulative evidence
11. higher selection confidence

### Tie-break rules
If two candidates remain tied:
1. prefer the candidate with `multi_horizon_confirmed`
2. prefer the candidate with more consecutive stable cycles
3. prefer the candidate with lower negative drift exposure
4. prefer the candidate with greater cumulative support
5. prefer the candidate with more recent consistency
6. if still tied, abstain rather than arbitrarily choose

## 11. Shadow Mode Operating Policy

Shadow mode is the required next phase after design-only.

### Shadow mode behavior
In shadow mode, the engine may:
- rank candidates
- produce a hypothetical selection
- emit abstain outcomes
- publish structured diagnostics
- write outputs for later review
- emit calibration-oriented confidence values for analysis only

In shadow mode, the engine must not:
- route to execution
- trigger orders
- change risk state
- affect capital deployment
- override downstream execution controls
- imply production readiness from a single-cycle output

### Shadow mode review goals
Shadow mode should be used to measure:
- selection frequency
- abstain frequency
- stability of top-ranked candidates over time
- drift sensitivity
- false confidence patterns
- disagreement between recent and cumulative evidence
- operational reliability of the schema and pipeline
- persistence quality of top-ranked candidates across multiple cycles

## 12. Example Shadow Output

```json
{
  "generated_at": "2026-03-15T00:00:00+00:00",
  "mode": "shadow",
  "selection_status": "abstain",
  "reason_codes": [
    "INSUFFICIENT_SAMPLE_SIZE",
    "NO_CANDIDATE_PASSED_STABILITY_GATE"
  ],
  "candidates_considered": 3,
  "selected_symbol": null,
  "selected_strategy": null,
  "selected_horizon": null,
  "selection_score": null,
  "selection_confidence": null,
  "ranking": [
    {
      "rank": 1,
      "symbol": "BTCUSDT",
      "strategy": "swing",
      "horizon": "4h",
      "candidate_status": "blocked",
      "selection_score": 2.8,
      "selection_confidence": 0.34,
      "reason_codes": [
        "DRIFT_BLOCKED",
        "WEAK_LATEST_SUPPORT"
      ],
      "latest_sample_size": 18,
      "cumulative_sample_size": 96,
      "symbol_cumulative_support": 180,
      "strategy_cumulative_support": 141,
      "selected_candidate_strength": "moderate",
      "selected_stability_label": "single_horizon_only",
      "consecutive_visible_cycles": 2,
      "consecutive_stable_cycles": 1,
      "drift_direction": "decrease",
      "score_delta": -0.5
    }
  ]
}
```

## 13. Future Activation Gates for Phase Transition

### Phase 1: Design-only
Requirements:
- schema defined
- reason-code taxonomy defined
- abstain policy defined
- ranking rules defined
- no execution linkage
- no capital allocation linkage
- no position state mutation
- no live authority granted to selection outputs

### Phase 2: Shadow mode
Requirements:
- schema-valid outputs produced consistently
- selection and abstain outcomes logged
- historical shadow outputs retained
- repeated review of drift behavior and false positives
- stable pipeline operation across multiple market conditions
- persistence metrics tracked across repeated cycles
- shadow outputs remain strictly non-executable
- operational review confirms that abstain behavior is conservative rather than overly permissive

### Phase 3: Limited activation
This phase must remain blocked until all of the following are satisfied:
- materially larger retained dataset than current level
- materially larger latest-window sample than current level
- shadow-mode evidence demonstrates stable abstain discipline
- drift handling proves conservative rather than reactive
- qualification criteria hold across multiple periods
- monitoring, alerting, and rollback controls are already in place
- explicit human review approves any activation step
- position state and capital controls exist outside the selection engine
- live activation remains capped to a narrow, reviewable scope
- selection output is advisory-to-gated, not autonomous-authoritative

### Phase 4: Capital-aware gated deployment
This phase must remain blocked until all of the following are satisfied:
- limited activation has already run without operational incidents across a meaningful observation period
- a separate capital allocation layer exists and is independently reviewable
- position lifecycle management exists and is independently testable
- exposure caps, symbol caps, and concurrent position caps are enforced outside the selection engine
- degraded-mode behavior is defined for upstream report failure, schema failure, and drift instability
- kill-switch and rollback procedures are documented and tested
- post-selection monitoring confirms that selected candidates do not exhibit unstable false-positive concentration

### Phase 5: Autonomous trading consideration
This phase must remain blocked until all of the following are satisfied:
- edge selection quality is stable across multiple market regimes
- capital allocation quality is stable across multiple market regimes
- position management is production-grade
- execution safeguards are production-grade
- test coverage is materially stronger than the current state
- schema validation and data-quality validation are enforced in production pathways
- operational observability is sufficient for rapid diagnosis and rollback
- human operators can fully explain why the system selected, abstained, reduced risk, or halted
- AI remains an interpreter layer and never becomes sole trading authority