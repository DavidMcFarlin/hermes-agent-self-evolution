# 🧬 Hermes Agent Self-Evolution

**Evolutionary self-improvement for [Hermes Agent](https://github.com/NousResearch/hermes-agent).**

Hermes Agent Self-Evolution uses DSPy + GEPA (Genetic-Pareto Prompt Evolution) to automatically evolve and optimize Hermes Agent's skills, tool descriptions, system prompts, and code — producing measurably better versions through reflective evolutionary search.

**No GPU training required.** Everything operates via API calls — mutating text, evaluating results, and selecting the best variants. ~$2-10 per optimization run.

## How It Works

```
Read current skill/prompt/tool ──► Generate eval dataset
                                        │
                                        ▼
                                   GEPA Optimizer ◄── Execution traces
                                        │                    ▲
                                        ▼                    │
                                   Candidate variants ──► Evaluate
                                        │
                                   Constraint gates (tests, size limits, benchmarks)
                                        │
                                        ▼
                                   Best variant ──► PR against hermes-agent
```

GEPA reads execution traces to understand *why* things fail (not just that they failed), then proposes targeted improvements. ICLR 2026 Oral, MIT licensed.

## Quick Start

```bash
# Install
git clone https://github.com/NousResearch/hermes-agent-self-evolution.git
cd hermes-agent-self-evolution
pip install -e ".[dev]"

# Point at your hermes-agent repo
export HERMES_AGENT_REPO=~/.hermes/hermes-agent

# Evolve a skill (synthetic eval data)
python -m evolution.skills.evolve_skill \
    --skill github-code-review \
    --iterations 10 \
    --eval-source synthetic

# Or use real session history from Claude Code, Copilot, and Hermes
python -m evolution.skills.evolve_skill \
    --skill github-code-review \
    --iterations 10 \
    --eval-source sessiondb

# Use the LLM-as-judge fitness metric (richer signal than keyword overlap)
# and deliver a passing improvement to a new branch in your hermes-agent repo:
python -m evolution.skills.evolve_skill \
    --skill github-code-review \
    --iterations 10 \
    --metric judge \
    --deliver           # add --open-pr to also push + open a PR via `gh`
```

### Fitness metrics

`--metric` selects how candidates are scored (skills and prompt sections):

| Value | Cost | What it measures |
|-------|------|------------------|
| `overlap` (default) | free | Keyword coverage vs. the rubric — fast inner-loop proxy |
| `judge` | LLM call | LLM-as-judge rates correctness/procedure on 0–1 with feedback |
| `hybrid` | LLM call | `judge` blended with a 0.3 keyword-overlap fallback |

GEPA always receives **score + feedback** (not a bare float) so its reflective
mutation has something to reason about.

## What It Optimizes

| Phase | Target | Engine | Status |
|-------|--------|--------|--------|
| **Phase 1** | Skill files (SKILL.md) | DSPy + GEPA | ✅ Implemented |
| **Phase 2** | Tool descriptions | DSPy + GEPA | ✅ Implemented |
| **Phase 3** | System prompt sections | DSPy + GEPA | ✅ Implemented |
| **Phase 4** | Tool implementation code | Darwinian Evolver / Internal | ✅ Implemented |
| **Phase 5** | Continuous improvement loop | Autonomous Orchestrator | ✅ Implemented (v1.0) |

## Engines

| Engine | What It Does | License |
|--------|-------------|---------|
| **[DSPy](https://github.com/stanfordnlp/dspy) + [GEPA](https://github.com/gepa-ai/gepa)** | Reflective prompt evolution — reads execution traces, proposes targeted mutations | MIT |
| **[Darwinian Evolver](https://github.com/imbue-ai/darwinian_evolver)** | Code evolution with Git-based organisms | AGPL v3 (external CLI only) |

## Guardrails

Every evolved variant must pass:
1. **Phase gate + holdout** — must show a real, gated improvement on a held-out split before it can be delivered
2. **Size & growth limits** — Skills ≤15KB, tool descriptions ≤500 chars, bounded growth over baseline
3. **Cost cap** — real per-model token spend is metered (`dspy.track_usage`) and checked against `max_cost_per_run_usd`; over-budget variants are not delivered
4. **Semantic preservation** — Must not drift from original purpose
5. **Full test suite on apply** — `--deliver` writes the variant into a throwaway `git worktree` and runs the hermes-agent test suite there; a failing suite discards the branch
6. **PR review** — delivery only ever creates a **branch**; pushing/opening a PR is opt-in (`--open-pr` via the `gh` CLI), so nothing reaches `main` without human review

## Delivery

Delivery closes the loop the diagram shows. It is layered so the
outward-facing parts are opt-in:

- `--deliver` — apply the evolved artifact to a new branch in `HERMES_AGENT_REPO`
  (local only), gated by the phase result, cost cap, and the repo's own test
  suite. Nothing is pushed.
- `--open-pr` — additionally push the branch and open a PR via `gh`.

Without `--deliver`, a run stops at `output/` exactly as before.

## Full Plan

See [PLAN.md](PLAN.md) for the complete architecture, evaluation data strategy, constraints, benchmarks integration, and phased timeline.

## License

MIT — © 2026 Nous Research
