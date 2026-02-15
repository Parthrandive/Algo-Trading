# Algo-Trading System

Multi-Agent AI Trading System for Indian Markets (NSE/F&O/MCX).

## Documentation

- **Master Plan**: [Multi_Agent_AI_Trading_System_Plan_Updated.md](docs/plans/Multi_Agent_AI_Trading_System_Plan_Updated.md)
- **Phase 1 Execution Plan**: [Phase_1_Data_Orchestration_Execution_Plan.md](docs/plans/Phase_1_Data_Orchestration_Execution_Plan.md)
- **Architecture Decisions**: [docs/architecture/](docs/architecture/)
- **Governance & Policies**: [docs/governance/](docs/governance/)

## Project Structure

- `src/`: Source code for agents and utilities.
- `tests/`: Unit and integration tests.
- `docs/`: Documentation, plans, and architectural records.
- `scripts/`: Utility scripts (e.g., smoke tests).
- `configs/`: Configuration files.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Tests**:
    ```bash
    pytest
    ```

3.  **Run Smoke Test**:
    ```bash
    python3 scripts/smoke_test.py
    ```
