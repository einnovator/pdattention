# PRA Control Plane

The Control Plane is the governed operations workspace for a PRA fleet. Its default desktop layout reserves 20% for navigation, 50% for the active workspace, and 30% for the PRA Agent.

## Start here

- Use **Fleet overview** to compare observed engine state with Registry intent.
- Select an engine in the left pane to inspect its runtime, model, sessions, resources, storage, and telemetry.
- Open [Registry and governance](registry.md) to manage approved models, bundles, profiles, deployments, and policies.
- Review [Audit and alerts](activity.md) before and after an operational change.
- Ask the [PRA Agent](agent.md) to summarize fleet state or locate relevant records.

## Workspace controls

Drag the divider beside the PRA Agent to resize chat. Drag the Dockview divider to resize navigation and the central workspace. The browser remembers both choices.

Use the avatar menu to switch between light and dark themes. Theme selection is stored locally in this browser.

## Access

Capabilities are role governed. Read-only operators can inspect fleet and Registry state. Mutating actions require the corresponding role, a reason, CSRF validation, and an audit record.

