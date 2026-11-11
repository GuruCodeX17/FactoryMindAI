
import re
from contextlib import AsyncExitStack

from mcp import Client

from src.factorymindai.agent.state import FactoryMindState
from src.factorymindai.agent.schemas import SupervisorDecision

from src.factorymindai.mcp_client import (
    create_server_parameters,
    call_mcp_tool
)

from src.factorymindai.rag.generator import (
    get_llm,
    generate_rag_answer
)

from src.factorymindai.services.prediction_service import (
    predict_machine_failure
)


# ============================================================
# 1. RAG NODE
# ============================================================


def rag_node(state: FactoryMindState) -> dict:
    """
    Retrieve maintenance knowledge using the original
    question and, when available, ML prediction context.
    """

    question = state.get("question", "").strip()
    top_k = state.get("top_k", 3)
    prediction = state.get("prediction", {})

    if not question:
        return {
            "error": "Question cannot be empty."
        }

    # Build prediction context when available
    rag_question = question

    if prediction:
        status = prediction.get(
            "status",
            "Not available"
        )

        probability = prediction.get(
            "failure_probability"
        )

        prediction_context = (
            "\n\nMachine prediction context:\n"
            f"Machine status: {status}\n"
        )

        if probability is not None:
            prediction_context += (
                f"Estimated failure probability: "
                f"{probability:.2%}\n"
            )

        prediction_context += (
            "Use this prediction context to retrieve "
            "relevant preventive maintenance guidance. "
            "Do not treat the prediction as a confirmed failure."
        )

        rag_question += prediction_context

    try:
        result = generate_rag_answer(
            question=rag_question,
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
    Use Groq LLM to classify user requests and select
    the appropriate agent or MCP tool.
    """

    question = state.get("question", "").strip()
    machine_data = state.get("machine_data")

    if not question:
        return {
            "intent": "unknown",
            "error": "Question cannot be empty."
        }

    # --------------------------------------------------------
    # Extract machine and part IDs using Python
    # --------------------------------------------------------

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
    # Groq LLM structured-output classification
    # --------------------------------------------------------

    try:
        llm = get_llm().with_structured_output(
            SupervisorDecision,
            method="json_schema",
            strict=False
        )

        prompt = f"""
        You are the supervisor of FactoryMindAI, an industrial
        predictive maintenance agent system.

        Classify the user's request.

        Available intents:
        rag, prediction, combined, maintenance,
        inventory, unknown.

        Available MCP tools:

        Maintenance server:
        - get_machine_status
        - get_maintenance_history

        Inventory server:
        - check_stock_availability
        - get_part_requirements
        - search_spare_parts

        Rules:
        1. Machine status questions must select
           get_machine_status.
        2. Maintenance history questions must select
           get_maintenance_history.
        3. Stock availability questions must select
           check_stock_availability.
        4. Spare part requirement questions must select
           get_part_requirements.
        5. Spare parts search questions must select
           search_spare_parts.
        6. For MCP requests, select the exact tool name.
        7. Never invent machine IDs or part IDs.
        8. For non-MCP requests, use mcp_tool="none".
        9. Do not create maintenance work orders.
        10. General troubleshooting questions should use RAG.
        11. If the user requests prediction and maintenance
            recommendations together, select combined.

        User question: {question}
        """

        decision = llm.invoke(prompt)

    except Exception as exc:
        return {
            "intent": "unknown",
            "error": f"Supervisor LLM error: {str(exc)}"
        }

    intent = decision.intent
    mcp_server = decision.mcp_server
    mcp_tool = decision.mcp_tool
    arguments = decision.arguments or {}

    question_lower = question.lower()

    # --------------------------------------------------------
    # Deterministic routing safeguards
    # --------------------------------------------------------

    prediction_keywords = [
        "predict",
        "prediction",
        "failure probability",
        "failure risk",
        "machine failure"
    ]

    recommendation_keywords = [
        "recommend",
        "recommendation",
        "preventive maintenance",
        "maintenance action",
        "what should i do",
        "what maintenance action",
        "troubleshoot",
        "how can i"
    ]

    needs_prediction = any(
        keyword in question_lower
        for keyword in prediction_keywords
    )

    needs_recommendation = any(
        keyword in question_lower
        for keyword in recommendation_keywords
    )

    # Combined workflow takes priority when sensor data exists
    if (
        needs_prediction
        and needs_recommendation
        and machine_data
    ):
        intent = "combined"
        mcp_tool = "none"
        mcp_server = "none"

    # General troubleshooting belongs to RAG
    elif (
        not machine_id
        and needs_recommendation
        and not needs_prediction
    ):
        intent = "rag"
        mcp_tool = "none"
        mcp_server = "none"

    # --------------------------------------------------------
    # Validate MCP tool selection
    # --------------------------------------------------------

    maintenance_tools = {
        "get_machine_status",
        "get_maintenance_history"
    }

    inventory_tools = {
        "check_stock_availability",
        "get_part_requirements",
        "search_spare_parts"
    }

    # Infer a tool if the LLM did not select one
    if mcp_tool == "none":

        if intent == "maintenance":

            if any(
                word in question_lower
                for word in [
                    "history",
                    "previous maintenance",
                    "service records"
                ]
            ):
                mcp_tool = "get_maintenance_history"

            else:
                mcp_tool = "get_machine_status"

        elif intent == "inventory":

            if any(
                word in question_lower
                for word in [
                    "stock",
                    "availability",
                    "quantity",
                    "in stock"
                ]
            ):
                mcp_tool = "check_stock_availability"

            elif any(
                word in question_lower
                for word in [
                    "requirement",
                    "replacement",
                    "repair"
                ]
            ):
                mcp_tool = "get_part_requirements"

            else:
                mcp_tool = "search_spare_parts"

    # --------------------------------------------------------
    # Maintenance MCP validation
    # --------------------------------------------------------

    if mcp_tool in maintenance_tools:

        if intent != "maintenance" or not machine_id:
            return {
                "intent": "unknown",
                "error": "A valid machine ID is required."
            }

        return {
            "intent": "maintenance",
            "mcp_server": "maintenance",
            "mcp_tool": mcp_tool,
            "mcp_arguments": {
                "machine_id": machine_id
            }
        }

    # --------------------------------------------------------
    # Inventory MCP validation
    # --------------------------------------------------------

    if mcp_tool in inventory_tools:

        if intent != "inventory":
            return {
                "intent": "unknown",
                "error": "Invalid inventory intent."
            }

        # Stock availability
        if mcp_tool == "check_stock_availability":

            if not part_id:
                return {
                    "intent": "unknown",
                    "error": (
                        "Please provide a part ID, "
                        "such as P-001."
                    )
                }

            quantity_match = re.search(
                r"\bquantity\s*(\d+)\b|"
                r"\b(\d+)\s*(?:units|pieces)\b",
                question,
                re.IGNORECASE
            )

            requested_quantity = 1

            if quantity_match:
                requested_quantity = int(
                    quantity_match.group(1)
                    or quantity_match.group(2)
                )

            if not 1 <= requested_quantity <= 10000:
                return {
                    "intent": "unknown",
                    "error": (
                        "Requested quantity must be "
                        "between 1 and 10000."
                    )
                }

            return {
                "intent": "inventory",
                "mcp_server": "inventory",
                "mcp_tool": mcp_tool,
                "mcp_arguments": {
                    "part_id": part_id,
                    "requested_quantity": requested_quantity
                }
            }

        # Required spare parts for maintenance
        if mcp_tool == "get_part_requirements":

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

            task = arguments.get("maintenance_task")

            # Verify the LLM's suggested task
            if task not in maintenance_tasks:

                task = next(
                    (
                        task_name
                        for task_name, phrases
                        in maintenance_tasks.items()
                        if any(
                            phrase in question_lower
                            for phrase in phrases
                        )
                    ),
                    None
                )

            if not task:
                return {
                    "intent": "unknown",
                    "error": (
                        "Please specify a supported "
                        "maintenance task."
                    )
                }

            return {
                "intent": "inventory",
                "mcp_server": "inventory",
                "mcp_tool": mcp_tool,
                "mcp_arguments": {
                    "maintenance_task": task
                }
            }

        # Spare parts search
        if mcp_tool == "search_spare_parts":

            search_query = arguments.get(
                "search_query",
                ""
            )

            if not isinstance(search_query, str):
                search_query = ""

            search_query = search_query.strip()[:100]

            return {
                "intent": "inventory",
                "mcp_server": "inventory",
                "mcp_tool": mcp_tool,
                "mcp_arguments": {
                    "search_query": search_query
                }
            }

    # --------------------------------------------------------
    # Reject unexpected MCP tools
    # --------------------------------------------------------

    if mcp_tool != "none":
        return {
            "intent": "unknown",
            "error": "The selected MCP tool is not allowed."
        }

    # --------------------------------------------------------
    # ML Prediction and RAG routing
    # --------------------------------------------------------

    if intent == "combined" and not machine_data:
        intent = "rag"

    if intent not in {"rag", "prediction", "combined"}:
        intent = "unknown"

    return {
        "intent": intent
    }


# ============================================================
# 4. MCP RESPONSE FORMATTER
# ============================================================

def format_mcp_response(
    tool_name: str,
    result: dict
) -> str:
    """
    Convert structured MCP tool results into readable responses.
    """

    if not isinstance(result, dict):
        return f"MCP tool completed. Result: {result}"

    # Machine status
    if tool_name == "get_machine_status":

        machine_id = result.get("machine_id", "Unknown")
        status = result.get("status", "Unknown")
        temperature = result.get(
            "temperature",
            "Not available"
        )
        vibration = result.get(
            "vibration",
            "Not available"
        )

        return (
            f"Machine {machine_id} is currently in {status} status.\n"
            f"Temperature: {temperature}\n"
            f"Vibration: {vibration}\n"
            "Please review these readings as part of the maintenance check."
        )

    # Maintenance history
    if tool_name == "get_maintenance_history":

        machine_id = result.get("machine_id", "Unknown")
        records = result.get(
            "history",
            result.get("records", [])
        )

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

        available_quantity = result.get(
            "available_quantity",
            0
        )

        requested_quantity = result.get(
            "requested_quantity",
            1
        )

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

        task = result.get(
            "maintenance_task",
            "Unknown"
        )

        parts = result.get(
            "parts",
            result.get("required_parts", [])
        )

        return (
            "Spare part requirements retrieved successfully.\n"
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

        # Convert MCP results into readable responses
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
