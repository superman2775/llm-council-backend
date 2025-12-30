# Railway Template Overview

> **Note**: This file is a Railway publishing asset, not user documentation.
> Use this content when creating/updating the Railway template marketplace listing.

---

## Tagline

Multi-model AI deliberation for higher-quality decisions

---

## Template Description

# Deploy and Host LLM Council on Railway

Stop relying on a single AI model. LLM Council orchestrates multiple large language models through a 3-stage deliberation process—parallel responses, anonymised peer review, and chairman synthesis—to produce more reliable, balanced answers than any single model alone.

## About Hosting LLM Council

LLM Council runs as a stateless HTTP server that coordinates queries across AI providers via OpenRouter. Deploy on Railway to get a dedicated API endpoint for integrating AI deliberation into your applications, automation workflows, or development tools.

The server handles:
- **Model coordination** across 4+ LLMs in parallel
- **Response anonymization** to eliminate model favoritism
- **Consensus building** with configurable verdict types
- **Webhook notifications** for async workflow integration

Build time is approximately 2-3 minutes. No database required—fully stateless.

## Common Use Cases

- **Workflow Automation** - Trigger council deliberations from webhooks, process results downstream. Perfect for decision gates in automated pipelines.
- **Automated Code Review** - Submit PRs for multi-model review. Get consensus recommendations with `binary` verdicts (approve/reject) or detailed `synthesis` feedback.
- **Support Ticket Triage** - Route tickets through the council for priority and category consensus. Multiple models cross-check to reduce misclassification.
- **Design Decision Validation** - Present architectural trade-offs to the council. Peer review ensures no single model's biases dominate.
- **Content Quality Gates** - Evaluate copy, documentation, or user content with multi-perspective analysis and aggregate scoring.

## Dependencies for LLM Council Hosting

- [OpenRouter](https://openrouter.ai) (default LLM provider)

Note: See also [Gateway Options](https://llm-council.dev/getting-started/configuration/#gateway-options) for alternative LLM providers and gateways.

### Deployment Dependencies

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | **Yes** | Your [OpenRouter](https://openrouter.ai) API key |
| `LLM_COUNCIL_API_TOKEN` | Recommended | Bearer token for API auth |

### Implementation Details

**Stack:** Python 3.11 with FastAPI, running as non-root user

**Health Check:** 
- (GET) /health returns {"status": "ok"}

**API Endpoints:**
- (POST) /v1/council/run — Synchronous deliberation
- (GET) /v1/council/stream — Server-Sent Events streaming

**Verdict Types:**
- `synthesis` — Natural language consensus (default)
- `binary` — Go/no-go with confidence score
- `tie_breaker` — Chairman resolves deadlocks

**Security:** All endpoints (except `/health`) require Bearer token authentication when `LLM_COUNCIL_API_TOKEN` is set.

### Getting Started

1. Click **Deploy** and connect your GitHub account
2. Add your `OPENROUTER_API_KEY` from [OpenRouter](https://openrouter.ai)
3. Generate an API token: `openssl rand -hex 16`
4. Set `LLM_COUNCIL_API_TOKEN` with your generated token
5. Deploy and test: curl https://your-app.railway.app/health

### Resources

- [Documentation](https://llm-council.dev/)
- [HTTP API Guide](https://llm-council.dev/guides/http-api/)
- [n8n Integration](https://llm-council.dev/integrations/n8n/)
- [GitHub Repository](https://github.com/amiable-dev/llm-council)

## Why Deploy LLM Council on Railway?

Railway is a singular platform to deploy your infrastructure stack. Railway will host your infrastructure so you don't have to deal with configuration, while allowing you to vertically and horizontally scale it.

By deploying LLM Council on Railway, you are one step closer to supporting a complete full-stack application with minimal burden. Host your servers, databases, AI agents, and more on Railway.

---

## Template Metadata

**Category:** AI / Machine Learning

**Tags:** ai, llm, deliberation, multi-model, consensus, api, workflow-automation, n8n

**Icon suggestion:** Brain or scales (representing deliberation/balance)

**GitHub Repo:** https://github.com/amiable-dev/llm-council
