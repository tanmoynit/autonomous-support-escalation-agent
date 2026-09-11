"""
Tools (function calling) available to the agent.

Guardrails live here, not in the prompt alone. The LLM can *ask* to process a
refund, but this code is the actual authority that decides whether a refund
is allowed. Even if the model is convinced to try, the business logic below
cannot be talked out of the auto-approval limit or the return window.
"""

from datetime import datetime
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from . import database as db
from .rag import get_retriever

# --- Guardrail constants -----------------------------------------------
AUTO_REFUND_LIMIT_USD = 200.0
RETURN_WINDOW_DAYS = 30


# --- Knowledge base search (RAG) ----------------------------------------
class KBSearchInput(BaseModel):
    query: str = Field(description="The customer's question or topic to search the policy knowledge base for.")


@tool("search_knowledge_base", args_schema=KBSearchInput)
def search_knowledge_base(query: str) -> str:
    """Search the company's policy knowledge base (returns, shipping,
    warranty, FAQ) for information relevant to the customer's question.
    Always use this before answering any question about policy - never rely
    on general knowledge for company-specific rules."""
    retriever = get_retriever(k=3)
    results = retriever.invoke(query)
    if not results:
        return "No relevant policy information found."
    return "\n\n---\n\n".join(
        f"[Source: {r.metadata.get('source', 'unknown')}]\n{r.page_content}" for r in results
    )


# --- Purchase verification ----------------------------------------------
class VerifyPurchaseInput(BaseModel):
    order_id: str = Field(description="The customer's order ID, e.g. ORD-1001")
    customer_email: str = Field(description="The email address associated with the order")


@tool("verify_purchase", args_schema=VerifyPurchaseInput)
def verify_purchase(order_id: str, customer_email: str) -> str:
    """Look up an order in the purchase history database and confirm it
    belongs to the given customer email. Always call this before discussing
    any specific order, and before any refund or escalation - never assume
    an order is valid without checking."""
    order = db.fetch_order_for_customer(order_id, customer_email)
    if not order:
        # Distinguish "doesn't exist" from "exists but wrong email" without
        # leaking other customers' data.
        exists = db.fetch_order(order_id)
        if exists:
            return f"No order '{order_id}' found for email '{customer_email}'. The order ID exists but does not match that email address."
        return f"No order found with ID '{order_id}'. Please double check the order ID."

    order_date = datetime.strptime(order["order_date"], "%Y-%m-%d")
    days_since = (datetime.utcnow() - order_date).days
    return (
        f"Order verified.\n"
        f"Order ID: {order['order_id']}\n"
        f"Product: {order['product_name']}\n"
        f"Amount: ${order['amount']:.2f}\n"
        f"Order date: {order['order_date']} ({days_since} days ago)\n"
        f"Already refunded: {'Yes' if order['refunded'] else 'No'}"
    )


# --- Return eligibility check --------------------------------------------
class EligibilityInput(BaseModel):
    order_id: str = Field(description="The order ID to check")
    customer_email: str = Field(description="The customer's email on the order")


@tool("check_return_eligibility", args_schema=EligibilityInput)
def check_return_eligibility(order_id: str, customer_email: str) -> str:
    """Check whether an order is eligible for an AUTOMATIC refund according
    to policy: within the 30-day return window, amount $200 or less, and not
    already refunded. Always call this before process_refund. If this
    returns not eligible, do not attempt process_refund - create an
    escalation ticket instead."""
    order = db.fetch_order_for_customer(order_id, customer_email)
    if not order:
        return "Cannot check eligibility: order not found for that customer."

    order_date = datetime.strptime(order["order_date"], "%Y-%m-%d")
    days_since = (datetime.utcnow() - order_date).days

    reasons_not_eligible = []
    if order["refunded"]:
        reasons_not_eligible.append("this order has already been refunded")
    if days_since > RETURN_WINDOW_DAYS:
        reasons_not_eligible.append(
            f"the order is {days_since} days old, past the {RETURN_WINDOW_DAYS}-day return window"
        )
    if order["amount"] > AUTO_REFUND_LIMIT_USD:
        reasons_not_eligible.append(
            f"the order amount (${order['amount']:.2f}) exceeds the ${AUTO_REFUND_LIMIT_USD:.2f} auto-approval limit"
        )

    if reasons_not_eligible:
        return (
            "NOT ELIGIBLE for automatic refund because: "
            + "; ".join(reasons_not_eligible)
            + ". This must be escalated to a human agent using create_escalation_ticket."
        )
    return "ELIGIBLE for automatic refund. You may proceed with process_refund."


# --- Refund processing (guarded) ------------------------------------------
class ProcessRefundInput(BaseModel):
    order_id: str = Field(description="The order ID to refund")
    customer_email: str = Field(description="The customer's email on the order")
    reason: str = Field(description="Brief reason the customer gave for the refund request")


@tool("process_refund", args_schema=ProcessRefundInput)
def process_refund(order_id: str, customer_email: str, reason: str) -> str:
    """Process an automatic refund for an order. This tool ENFORCES the
    guardrails itself (30-day window, $200 auto-approval limit, not already
    refunded) regardless of what has been discussed in conversation - it
    will refuse and tell you to escalate if the order does not qualify, even
    if you believe it should. Only call this after check_return_eligibility
    has confirmed the order is eligible."""
    order = db.fetch_order_for_customer(order_id, customer_email)
    if not order:
        return "REFUND DENIED: order not found for that customer. Cannot process."

    order_date = datetime.strptime(order["order_date"], "%Y-%m-%d")
    days_since = (datetime.utcnow() - order_date).days

    # Hard guardrail enforcement - independent of anything the model "decided".
    if order["refunded"]:
        return "REFUND DENIED: this order has already been refunded. Create an escalation ticket if the customer disputes this."
    if days_since > RETURN_WINDOW_DAYS:
        return (
            f"REFUND DENIED: order is {days_since} days old, past the {RETURN_WINDOW_DAYS}-day window. "
            "This cannot be auto-approved. Create an escalation ticket instead."
        )
    if order["amount"] > AUTO_REFUND_LIMIT_USD:
        return (
            f"REFUND DENIED: order amount ${order['amount']:.2f} exceeds the ${AUTO_REFUND_LIMIT_USD:.2f} "
            "auto-approval limit. This requires human review. Create an escalation ticket instead."
        )

    db.mark_refunded(order_id)
    return (
        f"REFUND APPROVED AND PROCESSED for order {order_id}. "
        f"${order['amount']:.2f} will be refunded to the original payment method "
        f"within 5-10 business days. Reason on file: {reason}"
    )


# --- Escalation ------------------------------------------------------------
class EscalationInput(BaseModel):
    order_id: Optional[str] = Field(default=None, description="Related order ID, if any")
    customer_email: str = Field(description="Customer's email address")
    issue_summary: str = Field(description="Short summary of the customer's issue")
    reason: str = Field(description="Why this needs human review (e.g. exceeds auto-refund limit, outside return window, warranty claim, customer dissatisfied)")


@tool("create_escalation_ticket", args_schema=EscalationInput)
def create_escalation_ticket(customer_email: str, issue_summary: str, reason: str, order_id: Optional[str] = None) -> str:
    """Create an escalation ticket for a human support agent to handle. Use
    this whenever a request cannot be safely or policy-compliantly resolved
    automatically: refunds outside auto-approval guardrails, warranty
    claims, shipping disputes, or whenever the customer explicitly asks for
    a human."""
    ticket_id = db.create_ticket(order_id, customer_email, issue_summary, reason)
    return (
        f"Escalation ticket #{ticket_id} created for {customer_email}. "
        f"A human support agent will follow up. Summary: {issue_summary} | Reason: {reason}"
    )


def get_tools():
    return [
        search_knowledge_base,
        verify_purchase,
        check_return_eligibility,
        process_refund,
        create_escalation_ticket,
    ]
