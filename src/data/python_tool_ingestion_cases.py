"""Ordinary Python callables used by Paper 6.5 zero-metadata ingestion tests."""

from __future__ import annotations

from typing import Literal, TypedDict


class UserRecord(TypedDict):
    user_id: str
    status: str


class RepositoryRecord(TypedDict):
    repository_id: str
    owner: str


class DocumentRecord(TypedDict):
    document_id: str
    text: str


class IdentifierResult(TypedDict):
    identifier: str


class ChangedResult(TypedDict):
    changed: bool


def search_user(email: str) -> str:
    """Find a user account by email address.

    Args:
        email: Email address associated with the account.

    Returns:
        Stable user identifier.
    """
    raise NotImplementedError


def get_user(user_id: str) -> UserRecord:
    """Retrieve one user account by its stable identifier.

    :param user_id: Stable user identifier.
    :return: The matching user record.
    """
    raise NotImplementedError


def validate_user(user_id: str) -> bool:
    """Check whether a user account may be changed.

    Parameters
    ----------
    user_id : str
        Stable user identifier.

    Returns
    -------
    bool
        Whether an update is allowed.
    """
    raise NotImplementedError


def update_user(user_id: str, status: str) -> ChangedResult:
    """Change the status associated with a user account.

    Args:
        user_id: Stable user identifier.
        status: New account status.

    Returns:
        Whether the account changed.
    """
    raise NotImplementedError


def notify_user(user_id: str, message: str) -> bool:
    """Send a message to a user about an account change.

    Args:
        user_id: Stable user identifier.
        message: Notification text.
    """
    raise NotImplementedError


def delete_user(user_id: str) -> bool:
    """Permanently remove a user account.

    :param user_id: Stable user identifier.
    :return: Whether the account was deleted.
    """
    raise NotImplementedError


def search_document(title: str) -> str:
    """Find a document by title.

    Args:
        title: Human-readable document title.

    Returns:
        Stable document identifier.
    """
    raise NotImplementedError


def read_document(document_id: str) -> DocumentRecord:
    """Read a document by identifier.

    Parameters
    ----------
    document_id : str
        Stable document identifier.
    """
    raise NotImplementedError


def extract_metadata(document_id: str) -> dict[str, str]:
    """Extract metadata fields from a document.

    Args:
        document_id: Stable document identifier.
    """
    raise NotImplementedError


def update_document(document_id: str, title: str) -> ChangedResult:
    """Change a document title.

    Args:
        document_id: Stable document identifier.
        title: Replacement title.
    """
    raise NotImplementedError


def export_document(document_id: str, format: Literal["pdf", "html", "text"]) -> str:
    """Export a document in a requested format.

    :param document_id: Stable document identifier.
    :param format: Output representation.
    :return: Identifier of the produced artifact.
    """
    raise NotImplementedError


def search_repository(name: str) -> str:
    """Find a source-code repository by name.

    Args:
        name: Repository name.
    """
    raise NotImplementedError


def get_repository(repository_id: str) -> RepositoryRecord:
    """Retrieve ownership details for a repository.

    Args:
        repository_id: Stable repository identifier.
    """
    raise NotImplementedError


def create_issue(repository_id: str, title: str) -> str:
    """Create a work-tracking issue in a repository.

    Args:
        repository_id: Stable repository identifier.
        title: Issue title.
    """
    raise NotImplementedError


def update_issue(issue_id: str, status: str) -> ChangedResult:
    """Change the status of a work-tracking issue.

    Args:
        issue_id: Stable issue identifier.
        status: New workflow status.
    """
    raise NotImplementedError


def create_report(artifact_id: str, title: str) -> str:
    """Create a report from an existing artifact.

    Args:
        artifact_id: Stable artifact identifier.
        title: Report title.
    """
    raise NotImplementedError


def archive_report(report_id: str) -> bool:
    """Move a completed report into the archive.

    :param report_id: Stable report identifier.
    """
    raise NotImplementedError


def purge_archive(artifact_id: str) -> bool:
    """Permanently remove an archived artifact.

    Args:
        artifact_id: Stable archived-artifact identifier.
    """
    raise NotImplementedError


PAPER6_5_TOOL_CALLABLES = (
    search_user,
    get_user,
    validate_user,
    update_user,
    notify_user,
    delete_user,
    search_document,
    read_document,
    extract_metadata,
    update_document,
    export_document,
    search_repository,
    get_repository,
    create_issue,
    update_issue,
    create_report,
    archive_report,
    purge_archive,
)
