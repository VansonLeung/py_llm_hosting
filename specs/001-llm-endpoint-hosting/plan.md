# Implementation Plan: LLM Endpoint Hosting

**Branch**: `001-llm-endpoint-hosting` | **Date**: 2025-11-05 | **Spec**: [link]
**Input**: Feature specification from `/specs/001-llm-endpoint-hosting/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Python console application for hosting LLM endpoints with OpenAI-compatible API, supporting chat completions (text and visual), embeddings, ranking, and full MCP protocol. Includes CLI for adding/removing servers persisted in JSON, optional API key auth, and modular architecture with callbacks for inter-modular communication.

## Technical Context

**Language/Version**: Python 3.11  
**Primary Dependencies**: FastAPI (REST API), Click (CLI), httpx (HTTP client), websockets, jsonrpc-async (MCP protocol), Pydantic (data models), Uvicorn (ASGI server)  
**Storage**: JSON file for server configurations  
**Testing**: pytest for unit and integration tests  
**Target Platform**: Cross-platform (Linux, macOS, Windows)  
**Project Type**: Single console application with embedded API server  
**Performance Goals**: API responses within 5s for chat completions, 2s for embeddings, 3s for ranking; support 50 concurrent requests  
**Constraints**: Modular code with callbacks for inter-modular communication, clean-to-read, separation of concerns. MCP implemented via JSON-RPC 2.0 over WebSocket for real-time tool integration.
**Callback Mechanism**: Use Python's asyncio event loop with custom event dispatcher for inter-modular communication. Implement callback registry in src/lib/callbacks.py with async event firing. Modules register callbacks via decorators; events trigger callbacks asynchronously to maintain separation of concerns.  
**Scale/Scope**: Local LLM hosting for up to 50 concurrent requests, multiple server management

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Code must be modularized and use callbacks for inter-modular communication
- Testing standards must be met with comprehensive test coverage
- User experience consistency ensured across all interfaces
- Performance requirements satisfied with benchmarks

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
src/
├── models/          # Data models (LLM Server, API Request, MCP Context)
├── services/        # Business logic (server management, API proxying)
├── cli/             # Console commands (add, remove, list servers)
├── api/             # REST API endpoints (OpenAI-compatible)
└── lib/             # Utilities (JSON persistence, callbacks)

tests/
├── contract/        # API contract tests
├── integration/     # End-to-end tests
└── unit/            # Unit tests for modules
```

**Structure Decision**: Single project structure chosen as it's a console application with embedded API, no separate frontend/backend needed.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
