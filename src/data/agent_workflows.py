"""Deterministic realistic tool workflows for Paper 6.5 M2--M5.

The catalog is intentionally small enough for repeated pretrained-model gates
but structurally rich enough to test discovery, typed execution, sequential
composition, and capability-neighborhood disclosure independently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping, Sequence

from pra_hf.agent_execution import SafeToolExecutor
from pra_hf.agent_resources import AgentResource, SideEffectClass, resource_uri


@dataclass(frozen=True)
class WorkflowStep:
    """One expected tool call and its deterministic result."""

    tool_name: str
    arguments: Mapping[str, object]
    output: Mapping[str, object]


@dataclass(frozen=True)
class WorkflowTask:
    """A held-out task with graded capability labels and an executable plan."""

    task_id: str
    family: str
    query: str
    steps: tuple[WorkflowStep, ...]
    useful_tools: frozenset[str] = frozenset()
    related_tools: frozenset[str] = frozenset()
    unsafe_tools: frozenset[str] = frozenset()

    @property
    def required_tools(self) -> tuple[str, ...]:
        return tuple(step.tool_name for step in self.steps)


def _tool(
    name: str,
    description: str,
    properties: Mapping[str, Mapping[str, object]],
    required: Sequence[str],
    *,
    category: str,
    object_type: str,
    operation_kind: str,
    returns: Mapping[str, object],
    consumes: Sequence[str] = (),
    produces: Sequence[str] = (),
    tags: Sequence[str] = (),
    side_effect: SideEffectClass = SideEffectClass.READ,
) -> AgentResource:
    """Create one typed tool with graph and schema provenance."""

    schema = {
        "name": name,
        "parameters": {
            "type": "object",
            "properties": dict(properties),
            "required": list(required),
            "additionalProperties": False,
        },
    }
    return AgentResource(
        uri=resource_uri("tool", "paper6_5", name, "v1"),
        kind="tool",
        namespace="paper6_5",
        name=name,
        version="v1",
        description=description,
        content=json.dumps(schema, sort_keys=True),
        aliases=(name.replace("_", " "),),
        side_effect_class=side_effect,
        tenant_id="paper6_5",
        metadata={
            "toolset_categories": (category,),
            "object_types": (object_type,),
            "api_family": f"paper6_5.{category}",
            "operation_family": object_type,
            "operation_kind": operation_kind,
            "provider": "paper6_5_fixture",
            "tags": tuple(tags),
            "consumes": tuple(consumes),
            "produces": tuple(produces),
            "returns": dict(returns),
        },
    )


def realistic_tool_catalog() -> tuple[AgentResource, ...]:
    """Return the frozen cross-domain catalog used by M2--M6."""

    string = {"type": "string"}
    boolean = {"type": "boolean"}
    return (
        _tool("search_user", "Find a user account by email address.", {"email": string}, ("email",), category="crm", object_type="user", operation_kind="search", returns={"user_id": "string"}, produces=("user_id",), tags=("user", "account", "search")),
        _tool("get_user", "Retrieve one user account by user identifier.", {"user_id": string}, ("user_id",), category="crm", object_type="user", operation_kind="get", returns={"user_id": "string", "status": "string"}, consumes=("user_id",), produces=("user_record",), tags=("user", "account", "read")),
        _tool("validate_user", "Validate whether a user account may be changed.", {"user_id": string}, ("user_id",), category="crm", object_type="user", operation_kind="validate", returns={"valid": "boolean"}, consumes=("user_id",), produces=("validation",), tags=("user", "validate")),
        _tool("update_user", "Update the status of a user account.", {"user_id": string, "status": string}, ("user_id", "status"), category="crm", object_type="user", operation_kind="update", returns={"changed": "boolean"}, consumes=("user_id", "validation"), produces=("user_change",), tags=("user", "account", "write"), side_effect=SideEffectClass.WRITE),
        _tool("notify_user", "Notify a user about an account change.", {"user_id": string, "message": string}, ("user_id", "message"), category="email", object_type="user", operation_kind="notify", returns={"notified": "boolean"}, consumes=("user_id", "user_change"), produces=("notification",), tags=("user", "notification"), side_effect=SideEffectClass.WRITE),
        _tool("delete_user", "Permanently delete a user account.", {"user_id": string}, ("user_id",), category="crm", object_type="user", operation_kind="delete", returns={"deleted": "boolean"}, consumes=("user_id",), tags=("user", "delete", "unsafe"), side_effect=SideEffectClass.DESTRUCTIVE),
        _tool("search_document", "Find a document by title.", {"title": string}, ("title",), category="documents", object_type="document", operation_kind="search", returns={"document_id": "string"}, produces=("document_id",), tags=("document", "search")),
        _tool("read_document", "Read a document by document identifier.", {"document_id": string}, ("document_id",), category="documents", object_type="document", operation_kind="read", returns={"document_id": "string", "text": "string"}, consumes=("document_id",), produces=("document_text",), tags=("document", "read")),
        _tool("extract_metadata", "Extract metadata from document text.", {"document_id": string}, ("document_id",), category="documents", object_type="document", operation_kind="inspect", returns={"metadata": "object"}, consumes=("document_id", "document_text"), produces=("document_metadata",), tags=("document", "metadata")),
        _tool("update_document", "Update a document title.", {"document_id": string, "title": string}, ("document_id", "title"), category="documents", object_type="document", operation_kind="update", returns={"changed": "boolean"}, consumes=("document_id",), produces=("document_change",), tags=("document", "write"), side_effect=SideEffectClass.WRITE),
        _tool("export_document", "Export a document in a requested format.", {"document_id": string, "format": string}, ("document_id", "format"), category="documents", object_type="document", operation_kind="export", returns={"artifact_id": "string"}, consumes=("document_id",), produces=("artifact_id",), tags=("document", "export")),
        _tool("search_repository", "Find a source repository by name.", {"name": string}, ("name",), category="github", object_type="repository", operation_kind="search", returns={"repository_id": "string"}, produces=("repository_id",), tags=("repository", "search")),
        _tool("get_repository", "Retrieve repository details.", {"repository_id": string}, ("repository_id",), category="github", object_type="repository", operation_kind="get", returns={"repository_id": "string", "owner": "string"}, consumes=("repository_id",), produces=("repository_record",), tags=("repository", "read")),
        _tool("create_issue", "Create a work-tracking issue in a repository.", {"repository_id": string, "title": string}, ("repository_id", "title"), category="github", object_type="issue", operation_kind="create", returns={"issue_id": "string"}, consumes=("repository_id",), produces=("issue_id",), tags=("repository", "issue", "write"), side_effect=SideEffectClass.WRITE),
        _tool("update_issue", "Update the status of a work-tracking issue.", {"issue_id": string, "status": string}, ("issue_id", "status"), category="github", object_type="issue", operation_kind="update", returns={"changed": "boolean"}, consumes=("issue_id",), produces=("issue_change",), tags=("issue", "write"), side_effect=SideEffectClass.WRITE),
        _tool("create_report", "Create a report from an artifact.", {"artifact_id": string, "title": string}, ("artifact_id", "title"), category="analytics", object_type="report", operation_kind="create", returns={"report_id": "string"}, consumes=("artifact_id",), produces=("report_id",), tags=("report", "artifact"), side_effect=SideEffectClass.WRITE),
        _tool("archive_report", "Archive a completed report.", {"report_id": string}, ("report_id",), category="analytics", object_type="report", operation_kind="archive", returns={"archived": "boolean"}, consumes=("report_id",), tags=("report", "archive"), side_effect=SideEffectClass.WRITE),
        _tool("purge_archive", "Permanently purge an archived artifact.", {"artifact_id": string}, ("artifact_id",), category="analytics", object_type="artifact", operation_kind="delete", returns={"purged": "boolean"}, consumes=("artifact_id",), tags=("artifact", "delete", "unsafe"), side_effect=SideEffectClass.DESTRUCTIVE),
    )


def workflow_tasks() -> tuple[WorkflowTask, ...]:
    """Return frozen one- through five-step tasks with graded labels."""

    return (
        WorkflowTask("m2-user", "user", "Retrieve user u17.", (WorkflowStep("get_user", {"user_id": "u17"}, {"user_id": "u17", "status": "active"}),), useful_tools=frozenset(("search_user",)), related_tools=frozenset(("validate_user", "update_user")), unsafe_tools=frozenset(("delete_user",))),
        WorkflowTask("m2-document", "document", "Read document d42.", (WorkflowStep("read_document", {"document_id": "d42"}, {"document_id": "d42", "text": "PRA notes"}),), useful_tools=frozenset(("search_document",)), related_tools=frozenset(("extract_metadata", "export_document"))),
        WorkflowTask("m2-repository", "repository", "Retrieve repository repo9.", (WorkflowStep("get_repository", {"repository_id": "repo9"}, {"repository_id": "repo9", "owner": "team-a"}),), useful_tools=frozenset(("search_repository",)), related_tools=frozenset(("create_issue",))),
        WorkflowTask("m2-export", "document", "Export document d42 as pdf.", (WorkflowStep("export_document", {"document_id": "d42", "format": "pdf"}, {"artifact_id": "artifact-d42-pdf"}),), useful_tools=frozenset(("read_document",)), related_tools=frozenset(("search_document",))),
        WorkflowTask("m4-user-3", "user", "Find alice@example.com, validate the account, then set its status to reviewed.", (WorkflowStep("search_user", {"email": "alice@example.com"}, {"user_id": "u17"}), WorkflowStep("validate_user", {"user_id": "u17"}, {"valid": True}), WorkflowStep("update_user", {"user_id": "u17", "status": "reviewed"}, {"changed": True})), useful_tools=frozenset(("get_user", "notify_user")), related_tools=frozenset(("delete_user",)), unsafe_tools=frozenset(("delete_user",))),
        WorkflowTask("m4-doc-4", "document", "Find the PRA Notes document, read it, export it as pdf, and create a report titled PRA digest.", (WorkflowStep("search_document", {"title": "PRA Notes"}, {"document_id": "d42"}), WorkflowStep("read_document", {"document_id": "d42"}, {"document_id": "d42", "text": "PRA notes"}), WorkflowStep("export_document", {"document_id": "d42", "format": "pdf"}, {"artifact_id": "artifact-d42-pdf"}), WorkflowStep("create_report", {"artifact_id": "artifact-d42-pdf", "title": "PRA digest"}, {"report_id": "report-7"})), useful_tools=frozenset(("extract_metadata",)), related_tools=frozenset(("update_document", "archive_report")), unsafe_tools=frozenset(("purge_archive",))),
        WorkflowTask("m4-repo-5", "repository", "Find repository pra-core, retrieve it, create issue Routing audit, mark that issue open, then notify user u17 about the change.", (WorkflowStep("search_repository", {"name": "pra-core"}, {"repository_id": "repo9"}), WorkflowStep("get_repository", {"repository_id": "repo9"}, {"repository_id": "repo9", "owner": "team-a"}), WorkflowStep("create_issue", {"repository_id": "repo9", "title": "Routing audit"}, {"issue_id": "issue-4"}), WorkflowStep("update_issue", {"issue_id": "issue-4", "status": "open"}, {"changed": True}), WorkflowStep("notify_user", {"user_id": "u17", "message": "Routing audit issue is open"}, {"notified": True})), useful_tools=frozenset(("get_user",)), related_tools=frozenset(("search_user",)), unsafe_tools=frozenset(("delete_user",))),
    )


def workflow_executor(resources: Sequence[AgentResource], task: WorkflowTask) -> SafeToolExecutor:
    """Build pure handlers whose outputs are fixed by the benchmark task."""

    outputs = {step.tool_name: dict(step.output) for step in task.steps}
    handlers = {}
    for resource in resources:
        result = outputs.get(resource.name, {"ok": True})
        handlers[resource.uri] = lambda _arguments, _observations, value=result: dict(value)
    return SafeToolExecutor(resources, handlers)

