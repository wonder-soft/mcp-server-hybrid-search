# Contributing to mcp-server-hybrid-search

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Create a feature branch from `main`
4. Make your changes
5. Submit a pull request

## Development Setup

### Prerequisites

- Rust toolchain (1.75+)
- Docker (for Qdrant)
- Python 3 + `markitdown` (`pip install markitdown`)
- OpenAI API key

### Build

```bash
cargo build
```

### Run Tests

```bash
cargo test
```

### Local Development

```bash
# Start Qdrant
docker compose up -d

# Set up environment
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# Initialize directories
cargo run -p ragctl -- init

# Run the MCP server
cargo run -p mcp-server-hybrid-search
```

## Coding Guidelines

### Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]
```

Types:
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `ci`: CI/CD changes

Scopes: `server`, `cli`, `common`, or omit for cross-cutting changes.

Examples:
```
feat(cli): add ragctl reset command
fix(server): handle UTF-8 boundary in snippet truncation
docs: update README with setup instructions
```

### Code Style

- Follow standard Rust conventions (`cargo fmt`, `cargo clippy`)
- Use `tracing` for logging (not `println!` in library code)
- Handle errors gracefully — prefer `Result` over `unwrap()`/`panic!()`
- Keep functions focused and small

### Pull Requests

- Create a branch from `main` with a descriptive name (e.g., `feat/add-pdf-support`)
- Keep PRs focused on a single concern
- Include a clear description of changes
- Ensure `cargo check` and `cargo test` pass before submitting

## Architecture

```
mcp-server-hybrid-search/
├── common/    # Shared types and configuration
├── server/    # MCP server (SSE, search tools)
├── cli/       # ragctl CLI (ingest, status, search, reset)
```

- **common**: Config loading (`config.toml`), shared types (`ChunkPayload`, `SearchResult`, etc.)
- **server**: Axum-based SSE MCP server with hybrid search (Qdrant + Tantivy + RRF)
- **cli**: Document ingestion pipeline with markitdown integration for PDF/Excel/Word

## Reporting Issues

- Use GitHub Issues
- Include steps to reproduce
- Include relevant logs and environment details (OS, Rust version, Qdrant version)

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
