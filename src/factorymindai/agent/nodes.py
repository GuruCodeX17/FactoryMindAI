import re
from contextlib import AsyncExitStack

from mcp import Client

from src.factorymindai.mcp_client import (
    create_server_parameters,
    call_mcp_tool
)
from src.factorymindai.agent.state import FactoryMindState
from src.factorymindai.rag.generator import generate_rag_answer


def rag_node(state: FactoryMindState) -> dict:
    """
    Retrieve maintenance knowledge and generate an answer.
    """

    question = state.get("question", "").strip()
    top_k = state.get("top_k", 3)

    if not question:
        return {
            "error": "Question cannot be empty."
        }

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

from src.factorymindai.services.prediction_service import (
    predict_machine_failure
)


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
                f"{result['failure_probability']:.2%}."
            )
        }

    except Exception as error:
        return {
            "error": f"ML prediction failed: {error}"
        }


def supervisor_node(state: FactoryMindState) -> dict:
    """
    Classify the user request and select the appropriate agent or MCP tool.
    """

    question = state.get("question", "").strip().lower()
    machine_data = state.get("machine_data")

    if not question:
        return {
            "intent": "unknown",
            "error": "Question cannot be empty."
        }

    # Extract machine ID and part ID
    machine_match = re.search(r"\bM-\d{3}\b", question, re.IGNORECASE)
    part_match = re.search(r"\bP-\d{3}\b", question, re.IGNORECASE)

    machine_id = machine_match.group().upper() if machine_match else None
    part_id = part_match.group().upper() if part_match else None

    # -----------------------------------
    # 1. MAINTENANCE MCP ROUTING
    # -----------------------------------

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

    # -----------------------------------
    # 2. INVENTORY MCP ROUTING
    # -----------------------------------

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

    # -----------------------------------
    # 3. ML PREDICTION ROUTING
    # -----------------------------------

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

async def mcp_node(state: FactoryMindState) -> dict:
    """
    Execute an MCP tool using the selected MCP server.
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

        return {
            "mcp_result": result,
            "final_answer": (
                f"MCP tool executed successfully: {tool_name}\n"
                f"Result: {result}"
            )
        }

    except Exception as error:
        return {
            "error": f"MCP execution failed: {error}"
        }