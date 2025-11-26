import logging

from fastapi import FastAPI, HTTPException

from .models import ToolBatchRequest, ToolBatchResponse
from .tool_executor import ToolExecutionError, execute_tool_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

app = FastAPI(title="Tool Service", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/tool-executions", response_model=ToolBatchResponse)
async def run_tool_batch(request: ToolBatchRequest) -> ToolBatchResponse:
    try:
        tool_messages, execution_log = await execute_tool_batch(
            request.tool_calls,
            request.tools,
        )
        return ToolBatchResponse(
            tool_messages=tool_messages,
            execution_log=execution_log,
        )
    except ToolExecutionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("tool_service.app.main:app", host="0.0.0.0", port=9001, reload=True)
