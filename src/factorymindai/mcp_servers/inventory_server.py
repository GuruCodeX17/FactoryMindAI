
from mcp.server import MCPServer

# Initialize Inventory MCP Server
mcp = MCPServer("FactoryMind Inventory")

# Sample spare-parts inventory
SPARE_PARTS = [
    {
        "part_id": "P-001",
        "part_name": "Ball Bearing",
        "category": "Mechanical",
        "quantity": 25,
        "minimum_stock": 5,
        "unit_price": 450.00
    },
    {
        "part_id": "P-002",
        "part_name": "Cooling Fan",
        "category": "Electrical",
        "quantity": 3,
        "minimum_stock": 5,
        "unit_price": 1200.00
    },
    {
        "part_id": "P-003",
        "part_name": "Hydraulic Pump",
        "category": "Hydraulic",
        "quantity": 0,
        "minimum_stock": 2,
        "unit_price": 8500.00
    },
    {
        "part_id": "P-004",
        "part_name": "Temperature Sensor",
        "category": "Electrical",
        "quantity": 15,
        "minimum_stock": 5,
        "unit_price": 750.00
    }
]

# Machine maintenance requirements
PART_REQUIREMENTS = {
    "bearing_replacement": ["P-001"],
    "cooling_system_repair": ["P-002"],
    "hydraulic_system_repair": ["P-003"],
    "temperature_sensor_replacement": ["P-004"]
}


# Tool 1: Search spare parts
@mcp.tool()
def search_spare_parts(search_query: str) -> dict:
    """Search spare parts using a part name or category."""

    query = search_query.lower()

    results = [
        part for part in SPARE_PARTS
        if query in part["part_name"].lower()
        or query in part["category"].lower()
    ]

    return {
        "search_query": search_query,
        "total_results": len(results),
        "parts": results
    }


# Tool 2: Check stock availability
@mcp.tool()
def check_stock_availability(
    part_id: str,
    requested_quantity: int = 1
) -> dict:
    """Check whether the requested quantity of a spare part is available."""

    if requested_quantity < 1:
        return {"error": "Requested quantity must be at least 1."}

    part = next(
        (item for item in SPARE_PARTS if item["part_id"] == part_id),
        None
    )

    if part is None:
        return {"error": f"Part {part_id} not found."}

    available = part["quantity"] >= requested_quantity

    return {
        "part_id": part_id,
        "part_name": part["part_name"],
        "available_quantity": part["quantity"],
        "requested_quantity": requested_quantity,
        "available": available
    }


# Tool 3: Get maintenance part requirements
@mcp.tool()
def get_part_requirements(maintenance_task: str) -> dict:
    """Find the spare parts required for a maintenance task."""

    task = maintenance_task.lower().strip()

    required_part_ids = PART_REQUIREMENTS.get(task)

    if required_part_ids is None:
        return {
            "error": f"No requirements found for task: {maintenance_task}"
        }

    required_parts = [
        part for part in SPARE_PARTS
        if part["part_id"] in required_part_ids
    ]

    return {
        "maintenance_task": maintenance_task,
        "required_parts": required_parts
    }


if __name__ == "__main__":
    mcp.run()
