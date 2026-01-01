# Multi-Model Deliberation: How LLM Council Verifies Code

*Published: January 2026*

---

Code review is hard. It requires understanding context, spotting subtle bugs, and making judgment calls about quality.

Your CI pipeline just approved a PR with a subtle TOCTOU race condition. One AI model said "looks good"—but it only caught 2 of 4 issues. A single model's opinion isn't enough for production code.

What if multiple AI models could deliberate together, anonymously evaluate each other's reviews, and reach a consensus verdict? That's exactly what LLM Council's verification system now does.

This post explains how 3-stage deliberation produces more reliable code verification than any single model alone.

## The Problem with Single-Model Verification

When you ask one AI to verify code, you get one opinion. That opinion might be:

- **Biased** by the model's training data
- **Overconfident** despite uncertainty
- **Inconsistent** across similar inputs
- **Blind** to certain vulnerability classes

Enterprise teams can't ship code based on a single model's "looks good to me." They need structured evaluation with transparent reasoning.

## The 3-Stage Deliberation Architecture

LLM Council verification runs three distinct stages, each designed to address single-model limitations:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Stage 1   │ ──► │   Stage 2   │ ──► │   Stage 3   │
│   Review    │     │  Peer Rank  │     │  Synthesis  │
└─────────────┘     └─────────────┘     └─────────────┘
     │                    │                    │
     ▼                    ▼                    ▼
  Multiple           Anonymous            Chairman
  parallel           evaluation           renders
  reviews            of reviews           verdict
```

### Stage 1: Parallel Model Reviews

Each council model independently reviews the code snapshot:

```python
stage1_results, _ = await stage1_collect_responses(verification_query)
# Returns: [
#   {"model": "openai/gpt-4o", "response": "...detailed review..."},
#   {"model": "anthropic/claude-3.5-sonnet", "response": "..."},
#   {"model": "google/gemini-pro-1.5", "response": "..."},
# ]
```

Why multiple models? Each has different:
- Training data and knowledge cutoffs
- Reasoning patterns and blind spots
- Sensitivity to different vulnerability types

One model might catch SQL injection while missing XSS. Another might spot race conditions but overlook CSRF. Together, they cover more ground.

### Stage 2: Anonymous Peer Ranking

Here's where it gets interesting. Each model evaluates the other reviews *without knowing who wrote them*:

```python
stage2_results, label_to_model, _ = await stage2_collect_rankings(
    verification_query, stage1_results
)
# Reviews presented as "Response A", "Response B", "Response C"
# Models rank them AND score on rubric dimensions
```

The anonymization is crucial. Without it, models might:
- Defer to "more prestigious" model names
- Self-promote their own responses
- Form cliques based on provider relationships

With anonymization, evaluation is based solely on review quality.

**Rubric Dimensions**: Each reviewer scores responses on:
- **Accuracy**: Are findings correct?
- **Relevance**: Do they address the actual code?
- **Completeness**: Are all issues identified?
- **Conciseness**: Is the review actionable?
- **Clarity**: Is the reasoning understandable?

### Stage 3: Chairman Synthesis

The chairman model synthesizes all reviews and rankings into a final verdict:

```python
stage3_result, _, _ = await stage3_synthesize_final(
    verification_query,
    stage1_results,
    stage2_results,
    aggregate_rankings=aggregate_rankings,
    verdict_type=VerdictType.BINARY,
)
# Returns: {"model": "...", "response": "...\nFINAL_VERDICT: APPROVED\n..."}
```

The chairman sees:
- All original reviews (**still anonymized** as "Response A", "Response B", etc.)
- All peer rankings (showing consensus)
- Aggregate Borda scores (ranking-based voting where 1st = N-1 points, 2nd = N-2, last = 0)

> **Note**: Model attribution is only revealed in the audit trail—the chairman makes decisions based solely on review quality, not model identity.

**Verdict Logic**: The chairman renders **APPROVED** or **REJECTED** (binary). The system then applies the confidence threshold:
- **PASS** (exit 0): APPROVED with confidence ≥ threshold (default 0.7)
- **FAIL** (exit 1): REJECTED
- **UNCLEAR** (exit 2): APPROVED but confidence below threshold, requires human review

## Dynamic Verdict Extraction

The raw synthesis needs structured extraction. That's where `verdict_extractor.py` comes in:

```python
import re
from statistics import stdev, mean

def calculate_confidence_from_agreement(stage2_results: list) -> float:
    """Calculate confidence from rubric score agreement (0.0-1.0)."""
    if not stage2_results:
        return 0.5

    # Collect all rubric scores across reviewers
    all_scores = []
    for result in stage2_results:
        scores = result.get("rubric_scores", {})
        all_scores.extend(scores.values())

    if len(all_scores) < 2:
        return 0.5

    # Low variance = high confidence (normalized to 0-1)
    # Max possible stdev for 1-10 scale is ~4.5
    variance_factor = 1.0 - min(stdev(all_scores) / 4.5, 1.0)
    return round(variance_factor, 2)

def extract_verdict_from_synthesis(stage3_result, stage2_results, threshold=0.7):
    """Extract verdict and confidence from chairman synthesis."""
    response = stage3_result.get("response", "")

    # Calculate confidence from reviewer agreement
    confidence = calculate_confidence_from_agreement(stage2_results)

    # Match FINAL_VERDICT with whitespace tolerance (handles model formatting variations)
    # Pattern allows leading whitespace and case-insensitive matching
    verdict_match = re.search(
        r"^\s*FINAL_VERDICT:\s*(APPROVED|REJECTED)\b",
        response,
        re.MULTILINE | re.IGNORECASE
    )

    if verdict_match:
        raw_verdict = verdict_match.group(1).upper()
        if raw_verdict == "APPROVED":
            # Apply confidence threshold for pass/unclear distinction
            if confidence >= threshold:
                return "pass", confidence
            return "unclear", confidence  # Low confidence triggers human review
        return "fail", confidence

    # No structured verdict found - default to unclear with low confidence
    return "unclear", 0.50
```

**Confidence calculation** is based on council agreement:
- **Rubric score variance**: Low variance across reviewers = high confidence
- **Ranking agreement**: Reviewers ranking responses similarly = high confidence
- **Borda count spread**: Clear winner (large point gap) = high confidence

## Audit Trail: Complete Transparency

Every verification writes a complete transcript:

```
.council/logs/2026-01-01T12-00-00-abc123/
├── request.json    # What was asked
├── stage1.json     # All individual reviews
├── stage2.json     # All peer rankings + rubric scores
├── stage3.json     # Chairman synthesis
└── result.json     # Final verdict + confidence
```

This enables:
- **Debugging**: Why did verification fail?
- **Auditing**: Who said what?
- **Learning**: How do models disagree?
- **Compliance**: Reproducible decisions

## Exit Codes for CI/CD

Verification returns machine-readable exit codes:

| Verdict | Exit Code | CI Action |
|---------|-----------|-----------|
| PASS | 0 | Continue pipeline |
| FAIL | 1 | Block deployment |
| UNCLEAR | 2 | Request human review |

Integration is straightforward:

```yaml
# GitHub Actions example
- name: Verify code changes
  env:
    PR_NUMBER: ${{ github.event.pull_request.number }}
  run: |
    # Capture exit code without failing immediately
    set +e
    llm-council verify ${{ github.sha }}
    exit_code=$?
    set -e

    case $exit_code in
      0) echo "Verification passed" ;;
      1) echo "Verification failed - blocking deployment"; exit 1 ;;
      2) gh pr comment "$PR_NUMBER" --body "Verification unclear - requesting human review"
         echo "Flagged for human review"; exit 0 ;;
    esac
```

## Performance Considerations

Multi-model deliberation trades speed for accuracy:

| Tier | Models | Typical Latency | Use Case |
|------|--------|-----------------|----------|
| quick | 2 | ~20-30s | Fast feedback, low-risk changes |
| balanced | 3 | ~45-60s | Standard PR verification |
| high | 4-5 | ~60-90s | Security-critical code |

While slower than single-model verification, the cost of a false positive ("looks good" on buggy code) far exceeds the cost of a 60-second deliberation. All three stages run in parallel where possible—Stage 1 queries all models simultaneously, and Stage 2 rankings are collected in parallel.

## What Makes This Different

Other AI verification approaches typically:
1. Use a single model (single point of failure)
2. Skip peer review (no quality check on reviews)
3. Return unstructured output (hard to automate)
4. Leave no audit trail (impossible to debug)

LLM Council verification provides:
- **Multi-model consensus** reducing individual bias
- **Anonymous peer review** ensuring quality evaluation
- **Structured verdicts** enabling automation
- **Complete transcripts** enabling transparency

## Try It

### Prerequisites

```bash
# Install the LLM Council MCP server
pip install llm-council-core

# Set your OpenRouter API key
export OPENROUTER_API_KEY="sk-or-..."
```

### Via MCP Client

```python
import asyncio
from mcp import ClientSession

async def verify_commit():
    async with ClientSession() as session:
        # Connect to the llm-council MCP server
        await session.connect("llm-council")

        result = await session.call_tool(
            "verify",
            {
                "snapshot_id": "abc1234",  # Git commit SHA
                "target_paths": ["src/"],  # Files/dirs to verify
                "rubric_focus": "Security",
                "confidence_threshold": 0.7
            }
        )
        print(result)

asyncio.run(verify_commit())
```

### Via CLI

```bash
# Verify a specific commit
llm-council verify abc1234 --paths src/ --focus security

# Verify HEAD with default settings
llm-council verify $(git rev-parse HEAD)
```

### Via Claude Code Skills

```bash
# In Claude Code, use the council-verify skill
/council-verify --snapshot HEAD~1 --focus security
```

---

*Multi-model deliberation isn't just about having multiple opinions—it's about structured evaluation, transparent reasoning, and reproducible decisions. That's what enterprise code verification requires.*
