---
name: pre-commit-test
description: Run tests and coverage check before committing. Use this skill whenever you're about to commit code and want to verify tests pass and see coverage metrics first. Catches broken commits early by running `make test` with a summary of failures and coverage report.
---

# Pre-Commit Test Check

A skill for running the test suite and coverage report before making a git commit, ensuring code quality and test coverage visibility.

## When to use

Run this skill before you commit code to catch test failures early and review coverage metrics.

## What it does

1. Executes `make test` to run the full test suite
2. Captures test output and extracts:
   - **Failed test count** — how many tests failed
   - **Coverage report** — coverage percentages by file/section
3. Stops immediately on first failure (non-zero exit code)
4. Displays a summary in the terminal

## Usage

When you're ready to commit and want to verify tests pass:

```
/pre-commit-test
```

The skill will:
- Run tests and report results
- Show which tests failed (if any)
- Display code coverage metrics
- Exit with status reflecting test outcome

If tests fail, fix the issues and run the skill again before committing.

## Implementation

Execute the following command in your project directory:

```bash
make test
```

This runs pytest on the test suite. The output will show:
- Test execution results (PASSED/FAILED)
- Number of failed tests (if any)
- Coverage report with percentages

Exit immediately if any test fails (captured by make's exit code).
