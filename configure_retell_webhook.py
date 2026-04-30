"""
One-time script to register the post-call webhook URL with Retell AI.

Usage:
    python configure_retell_webhook.py <YOUR_RAILWAY_URL>

Example:
    python configure_retell_webhook.py https://retell-efc-webhook-production.up.railway.app
"""

import sys
import os
import httpx

RETELL_API_KEY = os.environ.get("RETELL_API_KEY")
RETELL_BASE_URL = "https://api.retellai.com"
DEFAULT_AGENT_ID = "agent_27efcd8d33e3d52313d74a74a2"

POST_CALL_ANALYSIS_DATA = [
    {
        "name": "caller_name",
        "type": "string",
        "description": "The caller's full name if explicitly provided. Use N/A if not provided. Do not guess.",
    },
    {
        "name": "Callback Number",
        "type": "string",
        "description": "If the caller provides a callback number different from the calling number, extract that number. Otherwise use N/A.",
    },
    {
        "name": "Call_Type",
        "type": "enum",
        "choices": [
            "Trial Class",
            "Cancellation ",
            "Pausing Account",
            "General Question",
            "Spam or Hang up",
            "Lost Item",
            "Payment Error",
            "$99.00 Quickstart Sign up.",
            "Schedule ",
        ],
        "description": "The main category of the call.",
    },
    {
        "name": "Spam",
        "type": "enum",
        "choices": ["Yes", "No", "Maybe"],
        "description": "Yes if the caller is selling something unrelated, prank calling, or clearly irrelevant. Maybe if uncertain. Otherwise No.",
    },
    {
        "name": "trial_day",
        "type": "string",
        "description": "For backward compatibility only: the first active booked trial date in YYYY-MM-DD format. Use N/A unless final_outcome is booked or rescheduled. Never use weekday-only words, relative dates, or combined values.",
    },
    {
        "name": "trial_time",
        "type": "string",
        "description": "For backward compatibility only: the first active booked trial time, such as 7:00 PM. Use N/A unless final_outcome is booked or rescheduled. Do not combine multiple times.",
    },
    {
        "name": "call_date",
        "type": "string",
        "description": "The date of the call in YYYY-MM-DD format.",
    },
    {
        "name": "caller_phone",
        "type": "string",
        "description": "The caller's best callback phone number. Prefer a callback number stated by the caller; otherwise use the calling number. Use E.164 format when possible.",
    },
    {
        "name": "trial_booked",
        "type": "boolean",
        "description": "True only when final_outcome is booked or rescheduled and at least one trial class is still scheduled at the end of the call. False for cancelled, message, info_only, abandoned, and spam calls.",
    },
    {
        "name": "final_outcome",
        "type": "enum",
        "choices": ["booked", "rescheduled", "cancelled", "message", "info_only", "abandoned", "spam"],
        "description": "The final outcome of the call based on the caller's last expressed intent, not an earlier intent.",
    },
    {
        "name": "program",
        "type": "enum",
        "choices": ["Adult Jiu Jitsu", "Tiny Sharks", "Little Sharks", "Junior Sharks", "Multiple", "N/A"],
        "description": "Select the program for the active booked trial. Use Multiple only if active bookings are for different programs. Use N/A if no trial class remains booked.",
    },
    {
        "name": "trial_cancelled",
        "type": "boolean",
        "description": "True if the caller cancelled a trial or asked to cancel a trial during the call, even if a trial had been booked earlier in the conversation. False otherwise.",
    },
    {
        "name": "needs_follow_up",
        "type": "boolean",
        "description": "True if staff should follow up manually because the caller left a message, requested a callback, had a cancellation, had a payment issue, was frustrated, or the call ended before the caller's request was fully completed. False if no staff follow-up is needed.",
    },
    {
        "name": "follow_up_reason",
        "type": "string",
        "description": "Short reason staff should follow up. Use N/A if needs_follow_up is false.",
    },
    {
        "name": "bookings",
        "type": "string",
        "description": "JSON array of final active trial bookings only. Use [] if no trial remains booked. Include one object per booked student or class. Each object must include student_name, student_type, program, trial_date, trial_time, and status. Use status booked or rescheduled. trial_date must be YYYY-MM-DD. Do not include cancelled trials.",
    },
    {
        "name": "call_summary",
        "type": "system-presets",
        "description": "Write a 1-3 sentence summary of the call based on the call transcript. Include the final outcome.",
    },
    {
        "name": "call_successful",
        "type": "system-presets",
        "description": "Evaluate whether the agent completed the caller's final task without technical issues or caller frustration.",
    },
    {
        "name": "user_sentiment",
        "type": "system-presets",
        "description": "Evaluate user's sentiment, mood and satisfaction level.",
    },
]


def retell_headers():
    if not RETELL_API_KEY:
        raise RuntimeError("RETELL_API_KEY environment variable is required")
    return {
        "Authorization": f"Bearer {RETELL_API_KEY}",
        "Content-Type": "application/json",
    }


def list_agents():
    """List all agents to find the right agent_id."""
    resp = httpx.get(
        f"{RETELL_BASE_URL}/list-agents",
        headers=retell_headers(),
    )
    resp.raise_for_status()
    agents = resp.json()
    print(f"\nFound {len(agents)} agent(s):\n")
    for agent in agents:
        print(f"  ID: {agent.get('agent_id')}")
        print(f"  Name: {agent.get('agent_name', 'unnamed')}")
        print(f"  Webhook: {agent.get('post_call_analysis_data', 'none')}")
        print()
    return agents


def register_webhook(railway_url: str, target_agent_id: str = DEFAULT_AGENT_ID):
    """Register the post-call webhook URL with Retell AI."""
    webhook_url = f"{railway_url.rstrip('/')}/webhook/retell"

    # First, list agents
    agents = list_agents()

    if not agents:
        print("No agents found. Please create an agent in Retell AI first.")
        return

    matching_agents = [agent for agent in agents if agent.get("agent_id") == target_agent_id]
    if not matching_agents:
        print(f"No matching agent found for {target_agent_id}.")
        return

    for agent in matching_agents:
        agent_id = agent.get("agent_id")
        agent_name = agent.get("agent_name", "unnamed")

        print(f"Updating agent '{agent_name}' ({agent_id}) with webhook URL...")

        resp = httpx.patch(
            f"{RETELL_BASE_URL}/update-agent/{agent_id}",
            headers=retell_headers(),
            json={
                "post_call_analysis_data": POST_CALL_ANALYSIS_DATA,
                "webhook_url": webhook_url,
            },
        )
        resp.raise_for_status()

        print(f"  ✓ Agent '{agent_name}' updated successfully")
        print(f"    Webhook URL: {webhook_url}")

    print("\nDone! All agents configured.")
    print(f"Webhook endpoint: {webhook_url}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python configure_retell_webhook.py <YOUR_RAILWAY_URL>")
        print("Example: python configure_retell_webhook.py https://retell-efc-webhook-production.up.railway.app")
        sys.exit(1)

    register_webhook(sys.argv[1], os.environ.get("RETELL_AGENT_ID", DEFAULT_AGENT_ID))
