
from mcp.server import MCPServer

# Initialize the MCP server
mcp = MCPServer("FactoryMind Maintenance")

# Sample machine data (for development and testing)
MACHINES = {
    "M-001": {
        "machine_id": "M-001",
        "machine_type": "L",
        "status": "Running",
        "temperature": 78.5,
        "vibration": "Normal"
    },
    "M-002": {
        "machine_id": "M-002",
        "machine_type": "H",
        "status": "Warning",
        "temperature": 95.2,
        "vibration": "High"
    }
}

# Sample maintenance history
MAINTENANCE_HISTORY = {
    "M-001": [
        {
            "date": "2026-10-15",
            "activity": "Lubrication",
            "result": "Completed"
        }
    ],
    "M-002": [
        {
            "date": "2026-10-20",
            "activity": "Bearing inspection",
            "result": "Vibration detected"
        }
    ]
}

WORK_ORDERS = []


# TOOL 1: Get machine status
@mcp.tool()
def get_machine_status(machine_id: str) -> dict:
    """Get the current status and sensor information of a machine."""

    machine = MACHINES.get(machine_id)

    if machine is None:
        return {"error": f"Machine {machine_id} not found"}

    return machine


# TOOL 2: Get maintenance history
@mcp.tool()
def get_maintenance_history(machine_id: str) -> dict:
    """Retrieve previous maintenance records for a machine."""

    if machine_id not in MACHINES:
        return {"error": f"Machine {machine_id} not found"}

    history = MAINTENANCE_HISTORY.get(machine_id, [])

    return {
        "machine_id": machine_id,
        "maintenance_records": history
    }


# TOOL 3: Create maintenance work order
@mcp.tool()
def create_maintenance_work_order(
    machine_id: str,
    issue: str,
    approved: bool = False
) -> dict:
    """Create a demo work order only when approval is explicitly provided."""

    if machine_id not in MACHINES:
        return {"error": f"Machine {machine_id} not found"}

    if not approved:
        return {
            "status": "approval_required",
            "message": "Human approval is required before creating this work order."
        }

    work_order = {
        "work_order_id": f"WO-{len(WORK_ORDERS) + 1:03d}",
        "machine_id": machine_id,
        "issue": issue,
        "status": "Created"
    }

    WORK_ORDERS.append(work_order)

    return work_order


# Start the MCP server
if __name__ == "__main__":
    mcp.run()
