
import re
from contextlib import AsyncExitStack

from mcp import Client

from src.factorymindai.mcp_client import (
    create_server_parameters,
    call_mcp_tool
)
from src.factorymindai.agent.state import FactoryMindState
from src.factorymindai.rag.generator import generate_rag_answer
from src.factorymindai.services.prediction_service import (
    predict_machine_failure
)


# ============================================================
# 1. RAG NODE
# ============================================================

def rag_node(state: FactoryMindState) -> dict:
    """
    Retrieve maintenance knowledge and generate an answer.
    """

    question = state.get("question", "").strip()
    top_k = state.get("top_k", 3)

    if not question:
        return {"error": "Question cannot be empty."}

    try:
        result = generate_rag_answer(
            question=question,
            top_k=top_k
        )

        return {
            "rag_result": result,
            "sources": result.get("sources", []),
            "final_answer": result.get("answer", "")
        }

    except Exception as error:
        return {
            "error": f"RAG processing failed: {error}"
        }


# ============================================================
# 2. ML PREDICTION NODE
# ============================================================

def prediction_node(state: FactoryMindState) -> dict:
    """
    Predict machine failure using the trained ML model.
    """

    machine_data = state.get("machine_data")

    if not machine_data:
        return {
            "error": "Machine sensor data is required."
        }

    try:
        result = predict_machine_failure(machine_data)

        return {
            "prediction": result,
            "final_answer": (
                f"{result['status']}. "
                f"Failure probability: "
                f"{result['failure_probability']:.2%}. "
                "This is a model estimate, not a guarantee "
                "that the machine is safe."
            )
        }

    except Exception as error:
        return {
            "error": f"ML prediction failed: {error}"
        }


# ============================================================
# 3. SUPERVISOR NODE
# ============================================================

def supervisor_node(state: FactoryMindState) -> dict:
    """
    Classify the user request and select the appropriate
    agent or MCP tool.
    """

    question = state.get("question", "").strip().lower()
    machine_data = state.get("machine_data")

    if not question:
        return {
            "intent": "unknown",
            "error": "Question cannot be empty."
        }

    # Extract machine ID and part ID
    machine_match = re.search(
        r"\bM-\d{3}\b",
        question,
        re.IGNORECASE
    )

    part_match = re.search(
        r"\bP-\d{3}\b",
        question,
        re.IGNORECASE
    )

    machine_id = (
        machine_match.group().upper()
        if machine_match else None
    )

    part_id = (
        part_match.group().upper()
        if part_match else None
    )

    # --------------------------------------------------------
    # 3.1 MAINTENANCE MCP ROUTING
    # --------------------------------------------------------

    if any(word in question for word in [
        "machine status",
        "status of",
        "current status",
        "machine condition"
    ]):

        if not machine_id:
            return {
                "intent": "unknown",
                "error": "Please provide a machine ID, such as M-002."
            }

        return {
            "intent": "maintenance",
            "mcp_server": "maintenance",
            "mcp_tool": "get_machine_status",
            "mcp_arguments": {
                "machine_id": machine_id
            }
        }

    if any(word in question for word in [
        "maintenance history",
        "previous maintenance",
        "maintenance records",
        "service history"
    ]):

        if not machine_id:
            return {
                "intent": "unknown",
                "error": "Please provide a machine ID, such as M-002."
            }

        return {
            "intent": "maintenance",
            "mcp_server": "maintenance",
            "mcp_tool": "get_maintenance_history",
            "mcp_arguments": {
                "machine_id": machine_id
            }
        }

    # --------------------------------------------------------
    # 3.2 INVENTORY MCP ROUTING
    # --------------------------------------------------------

    if any(word in question for word in [
        "stock availability",
        "in stock",
        "stock of",
        "available quantity"
    ]):

        if not part_id:
            return {
                "intent": "unknown",
                "error": "Please provide a part ID, such as P-001."
            }

        quantity_match = re.search(
            r"\bquantity\s*(\d+)\b|\b(\d+)\s*(?:units|pieces)\b",
            question
        )

        requested_quantity = 1

        if quantity_match:
            requested_quantity = int(
                quantity_match.group(1) or quantity_match.group(2)
            )

        return {
            "intent": "inventory",
            "mcp_server": "inventory",
            "mcp_tool": "check_stock_availability",
            "mcp_arguments": {
                "part_id": part_id,
                "requested_quantity": requested_quantity
            }
        }

    # Maintenance task to required spare parts mapping
    maintenance_tasks = {
        "bearing_replacement": [
            "bearing replacement",
            "replace bearing"
        ],
        "cooling_system_repair": [
            "cooling system repair",
            "repair cooling system"
        ],
        "hydraulic_system_repair": [
            "hydraulic system repair",
            "repair hydraulic system"
        ],
        "temperature_sensor_replacement": [
            "temperature sensor replacement",
            "replace temperature sensor"
        ]
    }

    for task, phrases in maintenance_tasks.items():

        if any(phrase in question for phrase in phrases):

            return {
                "intent": "inventory",
                "mcp_server": "inventory",
                "mcp_tool": "get_part_requirements",
                "mcp_arguments": {
                    "maintenance_task": task
                }
            }

    # Spare parts search
    if any(word in question for word in [
        "spare parts",
        "inventory",
        "search parts",
        "find parts"
    ]):

        search_terms = [
            "electrical",
            "mechanical",
            "hydraulic",
            "ball bearing",
            "cooling fan",
            "hydraulic pump",
            "temperature sensor"
        ]

        search_query = next(
            (
                term for term in search_terms
                if term in question
            ),
            ""
        )

        return {
            "intent": "inventory",
            "mcp_server": "inventory",
            "mcp_tool": "search_spare_parts",
            "mcp_arguments": {
                "search_query": search_query
            }
        }

    # --------------------------------------------------------
    # 3.3 ML PREDICTION AND RAG ROUTING
    # --------------------------------------------------------

    prediction_keywords = [
        "predict",
        "prediction",
        "failure risk",
        "failure probability",
        "machine failure",
        "will fail",
        "sensor anomaly",
        "anomaly detection"
    ]

    rag_keywords = [
        "maintenance",
        "preventive",
        "troubleshoot",
        "recommend",
        "what should i do",
        "repair",
        "root cause",
        "diagnose",
        "equipment manual",
        "maintenance procedure"
    ]

    needs_prediction = any(
        keyword in question
        for keyword in prediction_keywords
    )

    needs_rag = any(
        keyword in question
        for keyword in rag_keywords
    )

    if needs_prediction and needs_rag and machine_data:
        intent = "combined"

    elif needs_prediction:
        intent = "prediction"

    elif needs_rag:
        intent = "rag"

    else:
        intent = "unknown"

    return {
        "intent": intent
    }


# ============================================================
# 4. MCP RESPONSE FORMATTER
# ============================================================

def format_mcp_response(tool_name: str, result: dict) -> str:
    """
    Convert structured MCP tool results into readable responses.
    """

    if not isinstance(result, dict):
        return f"MCP tool completed. Result: {result}"

    # Machine status
    if tool_name == "get_machine_status":

        machine_id = result.get("machine_id", "Unknown")
        status = result.get("status", "Unknown")
        temperature = result.get("temperature", "Not available")
        vibration = result.get("vibration", "Not available")

        return (
            f"Machine {machine_id} is currently in {status} status.\n"
            f"Temperature: {temperature}\n"
            f"Vibration: {vibration}\n"
            "Please review these readings as part of the maintenance check."
        )

    # Maintenance history
    if tool_name == "get_maintenance_history":

        machine_id = result.get("machine_id", "Unknown")
        records = result.get("history", result.get("records", []))

        return (
            f"Maintenance history retrieved for machine {machine_id}.\n"
            f"Records: {records if records else result}"
        )

    # Stock availability
    if tool_name == "check_stock_availability":

        part_name = result.get(
            "part_name",
            result.get("part_id", "Unknown part")
        )

        available_quantity = result.get("available_quantity", 0)
        requested_quantity = result.get("requested_quantity", 1)
        available = result.get("available", False)

        if available:
            return (
                f"{part_name} is available in stock.\n"
                f"Available quantity: {available_quantity}\n"
                f"Requested quantity: {requested_quantity}"
            )

        return (
            f"Insufficient stock for {part_name}.\n"
            f"Available quantity: {available_quantity}\n"
            f"Requested quantity: {requested_quantity}\n"
            "Please check inventory or arrange replenishment."
        )

    # Spare parts search
    if tool_name == "search_spare_parts":

        parts = result.get("parts", [])

        if not parts:
            return "No matching spare parts were found."

        lines = ["Matching spare parts found:"]

        for part in parts:
            lines.append(
                f"- {part.get('part_name', 'Unknown part')} "
                f"(ID: {part.get('part_id', 'Unknown')}) | "
                f"Quantity: {part.get('quantity', 'Unknown')} | "
                f"Price: INR {part.get('unit_price', 'Unknown')}"
            )

        return "\n".join(lines)

    # Required parts for maintenance tasks
    if tool_name == "get_part_requirements":

        task = result.get("maintenance_task", "Unknown")
        parts = result.get("parts", result.get("required_parts", []))

        return (
            f"Spare part requirements retrieved successfully.\n"
            f"Maintenance task: {task}\n"
            f"Required parts: {parts if parts else result}"
        )

    # Generic fallback
    lines = ["MCP tool executed successfully."]

    for key, value in result.items():
        readable_key = key.replace("_", " ").title()
        lines.append(f"{readable_key}: {value}")

    return "\n".join(lines)


# ============================================================
# 5. MCP NODE
# ============================================================

async def mcp_node(state: FactoryMindState) -> dict:
    """
    Execute an MCP tool and generate a readable response.
    """

    server_name = state.get("mcp_server")
    tool_name = state.get("mcp_tool")
    arguments = state.get("mcp_arguments", {})

    if server_name not in ["maintenance", "inventory"]:
        return {
            "error": "Invalid MCP server selected."
        }

    if not tool_name:
        return {
            "error": "No MCP tool selected."
        }

    try:
        server_filename = f"{server_name}_server.py"

        server_params = create_server_parameters(
            server_filename
        )

        async with AsyncExitStack() as stack:

            client = await stack.enter_async_context(
                Client(server_params)
            )

            clients = {
                server_name: client
            }

            result = await call_mcp_tool(
                clients=clients,
                server_name=server_name,
                tool_name=tool_name,
                arguments=arguments
            )

        # Convert MCP result into a readable response
        formatted_answer = format_mcp_response(
            tool_name=tool_name,
            result=result
        )

        return {
            "mcp_result": result,
            "final_answer": formatted_answer
        }

    except Exception as error:
        return {
            "error": f"MCP execution failed: {error}"
        }
