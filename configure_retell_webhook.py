"""
One-time script to register the post-call webhook URL with Retell AI.

Usage:
    python configure_retell_webhook.py <YOUR_RAILWAY_URL>

Example:
    python configure_retell_webhook.py https://retell-efc-webhook-production.up.railway.app
"""

import sys
import httpx

RETELL_API_KEY = "key_dc76bd1b9f12d1e105c211c337ea"
RETELL_BASE_URL = "https://api.retellai.com"


def list_agents():
    """List all agents to find the right agent_id."""
    resp = httpx.get(
        f"{RETELL_BASE_URL}/list-agents",
        headers={"Authorization": f"Bearer {RETELL_API_KEY}"},
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


def register_webhook(railway_url: str):
    """Register the post-call webhook URL with Retell AI."""
    webhook_url = f"{railway_url.rstrip('/')}/webhook/retell"

    # First, list agents
    agents = list_agents()

    if not agents:
        print("No agents found. Please create an agent in Retell AI first.")
        return

    # Register webhook for each agent
    for agent in agents:
        agent_id = agent.get("agent_id")
        agent_name = agent.get("agent_name", "unnamed")

        print(f"Updating agent '{agent_name}' ({agent_id}) with webhook URL...")

        resp = httpx.patch(
            f"{RETELL_BASE_URL}/update-agent/{agent_id}",
            headers={
                "Authorization": f"Bearer {RETELL_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "post_call_analysis_data": [
                    {
                        "name": "first_name",
                        "type": "string",
                        "description": "The caller's first name",
                    },
                    {
                        "name": "last_name",
                        "type": "string",
                        "description": "The caller's last name",
                    },
                    {
                        "name": "phone",
                        "type": "string",
                        "description": "The caller's phone number",
                    },
                    {
                        "name": "trial_booked",
                        "type": "boolean",
                        "description": "Whether a trial class was booked during the call",
                    },
                    {
                        "name": "trial_class_day",
                        "type": "string",
                        "description": "The day of the week the trial class is booked for",
                    },
                    {
                        "name": "trial_class_time",
                        "type": "string",
                        "description": "The time the trial class is booked for",
                    },
                ],
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

    register_webhook(sys.argv[1])
