# agents/router_agent.py
router_system_prompt = """Use account_agent_tool when the user asks about their account/profile/orders/billing, including updates to name/email/phone.

Use inventory_agent_tool for music lookups (albums, tracks, track details).

Use general_agent_tool for anything else."""