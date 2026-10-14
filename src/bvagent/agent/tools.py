"""Tool schemas and handlers for the two Verbex-style scenarios.

Schemas are written for a voice channel, which changes what a good tool looks
like. Descriptions tell the model to ask for one thing at a time, because a
caller cannot read a form; parameters stay few and flat, because every argument
the model has to invent is a chance to stall mid-call while it reasons.
"""

from __future__ import annotations

from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams

from bvagent.agent.backend import Backend


def build_tools(backend: Backend, scenario: str) -> ToolsSchema:
    """Tool set for ``scenario``: ``"loan"`` or ``"booking"``.

    Only the scenario's own tools are advertised. Offering both would put
    unused schemas in every prompt, which costs prefill time on every turn —
    latency the caller pays for a capability this call will never use.
    """
    if scenario == "loan":
        return ToolsSchema(standard_tools=_loan_tools(backend))
    if scenario == "booking":
        return ToolsSchema(standard_tools=_booking_tools(backend))
    raise ValueError(f"unknown scenario {scenario!r}; expected 'loan' or 'booking'")


def _handler(backend: Backend, method: str, *arg_names: str):
    """Adapt a backend coroutine to Pipecat's function-call protocol."""

    async def run(params: FunctionCallParams) -> None:
        kwargs = {name: params.arguments.get(name) for name in arg_names}
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        logger.info(f"tool {method}({kwargs})")
        try:
            result = await getattr(backend, method)(**kwargs)
        except TypeError as exc:
            # The model invented an argument or omitted a required one. Telling
            # it so in-band beats failing the turn: it can ask the caller.
            logger.warning(f"tool {method} rejected arguments {kwargs}: {exc}")
            result = {"error": "প্রয়োজনীয় তথ্য পাইনি।", "detail": str(exc)}
        await params.result_callback(result)

    run.__name__ = method
    return run


def _loan_tools(backend: Backend) -> list[FunctionSchema]:
    return [
        FunctionSchema(
            name="get_loan_status",
            description=(
                "Look up a customer's loan: installment amount, total outstanding, "
                "due date, and whether it is overdue. Requires the account number. "
                "If the caller has not given one, ask for it first — do not guess."
            ),
            properties={
                "account_number": {
                    "type": "string",
                    "description": "Loan account number, digits only. Accepts Bengali or ASCII numerals.",
                }
            },
            required=["account_number"],
            handler=_handler(backend, "get_loan_status", "account_number"),
        ),
        FunctionSchema(
            name="record_payment_promise",
            description=(
                "Record that the customer promised to pay by a date. Call this only "
                "after the customer has actually stated a date. Confirm the date back "
                "to them in the same turn."
            ),
            properties={
                "account_number": {"type": "string", "description": "Loan account number."},
                "promised_date": {
                    "type": "string",
                    "description": "The date the customer said, in their own words (e.g. 'আগামী রবিবার').",
                },
                "amount_bdt": {
                    "type": "integer",
                    "description": "Amount in taka, if the customer named one. Omit to use the installment amount.",
                },
            },
            required=["account_number", "promised_date"],
            handler=_handler(
                backend, "record_payment_promise", "account_number", "promised_date", "amount_bdt"
            ),
        ),
    ]


def _booking_tools(backend: Backend) -> list[FunctionSchema]:
    return [
        FunctionSchema(
            name="find_available_slots",
            description=(
                "List open appointment slots at a branch. Call this as soon as the "
                "caller names a branch; do not ask for a preferred day first."
            ),
            properties={
                "branch": {
                    "type": "string",
                    "description": "Branch name in Bangla, e.g. গুলশান, ধানমন্ডি, উত্তরা, চট্টগ্রাম.",
                },
                "preferred_day": {
                    "type": "string",
                    "description": "Optional day the caller mentioned.",
                },
            },
            required=["branch"],
            handler=_handler(backend, "find_available_slots", "branch", "preferred_day"),
        ),
        FunctionSchema(
            name="book_appointment",
            description=(
                "Book one of the slots returned by find_available_slots. Use the "
                "slot_id exactly as given by that tool, not the spoken description."
            ),
            properties={
                "customer_name": {"type": "string", "description": "Caller's name."},
                "branch": {"type": "string", "description": "Branch name."},
                "slot_id": {
                    "type": "string",
                    "description": "slot_id from find_available_slots (ISO timestamp).",
                },
            },
            required=["customer_name", "branch", "slot_id"],
            handler=_handler(backend, "book_appointment", "customer_name", "branch", "slot_id"),
        ),
    ]
