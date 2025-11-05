# Support Bot

Routes customer inquiries to specialized agents for account management, music inventory lookups, and general support.

## Architecture

The bot uses a supervisor agent pattern with three specialized agents:

- **Account Agent**: Handles customer account information, profile updates, and invoice lookups
- **Inventory Agent**: Handles music catalog queries (albums, tracks, track details)
- **General Support Agent**: Handles general inquiries and non-specific questions

The workflow includes automatic conversation summarization when the message count exceeds a threshold.

## Features

- Multi-agent routing for different query types
- Customer account management (view and edit profile information)
- Music inventory search and lookup
- Conversation summarization for long sessions
- Prompt injection detection and prevention
- PII detection and masking
- Human-in-the-loop approvals for sensitive operations
- Tool error handling and retry logic

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with the following variables:
```
OPENAI_API_KEY=your_openai_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
```

## Usage

The main entry point is `agent.py`. The compiled graph can be invoked programmatically or used with LangGraph Studio.

The bot expects a customer ID to be provided in the conversation context. It uses an in-memory SQLite database (Chinook database) for demonstration purposes.

## Project Structure

```
support-bot/
├── agent.py                 # Main agent workflow and supervisor
├── agents/
│   ├── router_agent.py      # Router/supervisor agent prompt
│   ├── customer_agent.py    # Account management agent and tools
│   ├── music_agent.py       # Music inventory agent and tools
│   └── general_support.py   # General support agent prompt
├── utils/
│   ├── database.py          # Database connection and setup
│   ├── model.py             # Shared LLM model configuration
│   ├── contexts.py          # State schemas for agents
│   └── prompt_injection.py  # Prompt injection guard middleware
├── requirements.txt         # Python dependencies
└── langgraph.json          # LangGraph Studio configuration
```

## Agents and Tools

### Account Agent
- `get_customer_info`: Retrieves customer information
- `edit_customer_info`: Updates customer address, phone, or email (requires human approval)
- `past_invoices`: Looks up past invoices for a customer

### Inventory Agent
- `get_albums_by_artist`: Searches albums by artist name
- `get_tracks_by_artist`: Searches tracks by artist name
- `get_info_about_track`: Gets detailed information about a specific track

### General Support Agent
- No tools, handles general inquiries conversationally

## Middleware

The agents use various middleware for security and reliability:

- **Prompt Injection Guard**: Detects and blocks prompt injection attempts
- **PII Middleware**: Masks email addresses in outputs
- **Human-in-the-Loop**: Requires approval before editing customer information
- **Tool Retry Middleware**: Automatically retries failed tool calls with exponential backoff
- **Tool Error Handler**: Gracefully handles tool errors without breaking the workflow

## State Management

The bot uses LangGraph's state management with checkpoints stored in-memory. The state includes:
- Messages (conversation history)
- Customer ID
- Username
- Summary (for long conversations)

## Notes

- The database is an in-memory SQLite instance populated with the Chinook sample database
- Conversation summarization occurs after 4 messages to manage context length
- Only Address, Phone, and Email can be edited for customer accounts
