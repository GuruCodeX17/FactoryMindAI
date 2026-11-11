
import json
import sys
import asyncio

from pathlib import Path
from contextlib import AsyncExitStack

from mcp import Client, StdioServerParameters


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PYTHON_EXECUTABLE = sys.executable


# ==========================================
# 2. CREATE MCP SERVER PARAMETERS
# ==========================================

def create_server_parameters(server_filename: str):
    """Create subprocess configuration for an MCP server."""

    server_path = (
        PROJECT_ROOT
        / "src"
        / "factorymindai"
        / "mcp_servers"
        / server_filename
    )

    return StdioServerParameters(
        command=PYTHON_EXECUTABLE,
        args=[str(server_path)]
    )


# ==========================================
# 3. PARSE MCP TOOL RESPONSE
# ==========================================

def parse_tool_result(tool_result):
    """Convert an MCP tool response into Python data."""

    if tool_result.is_error:
        raise RuntimeError(
            f"MCP tool execution failed: {tool_result.content}"
        )

    # Prefer structured data when available
    if tool_result.structured_content is not None:
        return tool_result.structured_content

    # Extract JSON from text content
    text_content = next(
        (
            item.text
            for item in tool_result.content
            if item.type == "text"
        ),
        None
    )

    if text_content is None:
        raise ValueError(
            "MCP tool returned no usable content."
        )

    return json.loads(text_content)


# ==========================================
# 4. REUSABLE MCP TOOL CALLING FUNCTION
# ==========================================

async def call_mcp_tool(
    clients: dict,
    server_name: str,
    tool_name: str,
    arguments: dict
):
    """Execute an MCP tool from the selected server."""

    # Select the MCP client
    client = clients.get(server_name)

    if client is None:
        raise ValueError(
            f"MCP server '{server_name}' not found."
        )

    # Execute the requested tool
    tool_result = await client.call_tool(
        tool_name,
        arguments
    )

    # Parse and return the response
    return parse_tool_result(tool_result)


# ==========================================
# 5. MAIN APPLICATION
# ==========================================

async def main():

    async with AsyncExitStack() as stack:

        # --------------------------------------
        # Connect to Maintenance MCP Server
        # --------------------------------------

        maintenance_params = create_server_parameters(
            "maintenance_server.py"
        )

        maintenance_client = await stack.enter_async_context(
            Client(maintenance_params)
        )

        # --------------------------------------
        # Connect to Inventory MCP Server
        # --------------------------------------

        inventory_params = create_server_parameters(
            "inventory_server.py"
        )

        inventory_client = await stack.enter_async_context(
            Client(inventory_params)
        )

        print("\nBoth MCP servers connected successfully!")

        # --------------------------------------
        # Register MCP clients
        # --------------------------------------

        clients = {
            "maintenance": maintenance_client,
            "inventory": inventory_client
        }

        # --------------------------------------
        # Discover Maintenance MCP tools
        # --------------------------------------

        maintenance_tools = await maintenance_client.list_tools()

        print("\n========== MAINTENANCE MCP TOOLS ==========")

        for tool in maintenance_tools.tools:
            print(f"- {tool.name}")

        # --------------------------------------
        # Discover Inventory MCP tools
        # --------------------------------------

        inventory_tools = await inventory_client.list_tools()

        print("\n========== INVENTORY MCP TOOLS ==========")

        for tool in inventory_tools.tools:
            print(f"- {tool.name}")

        # ======================================
        # 6. EXECUTE MAINTENANCE MCP TOOL
        # ======================================

        print("\n========== MAINTENANCE TOOL EXECUTION ==========")

        machine_status = await call_mcp_tool(
            clients,
            "maintenance",
            "get_machine_status",
            {
                "machine_id": "M-002"
            }
        )

        print("\nMachine status result:")
        print(machine_status)

        print("\nMachine status:", machine_status["status"])
        print("Temperature:", machine_status["temperature"])
        print("Vibration:", machine_status["vibration"])

        # ======================================
        # 7. EXECUTE INVENTORY MCP TOOL
        # ======================================

        print("\n========== INVENTORY TOOL EXECUTION ==========")

        inventory_data = await call_mcp_tool(
            clients,
            "inventory",
            "search_spare_parts",
            {
                "search_query": "electrical"
            }
        )

        print("\nInventory search result:")
        print(inventory_data)

        print("\nAvailable parts:")

        for part in inventory_data.get("parts", []):
            print(
                f"{part['part_name']} | "
                f"Quantity: {part['quantity']} | "
                f"Price: INR {part['unit_price']}"
            )

        print("\n========== MCP EXECUTION COMPLETED ==========")


# ==========================================
# 8. APPLICATION ENTRY POINT
# ==========================================

if __name__ == "__main__":
    asyncio.run(main())
