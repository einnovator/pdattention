"""Curated canonical concepts for Paper 6.5 semantic-hard discovery."""

from __future__ import annotations

from pra_hf.semantic_resource_discovery import CanonicalConceptMap, ConceptExpansion


_EN_OPERATIONS = {
    "search": ("search", "find", "locate", "look up", "track down", "identify", "resolve", "where is"),
    "get": ("get", "retrieve", "show", "pull up", "load", "display", "what do we know"),
    "validate": ("validate", "verify", "confirm", "check eligibility", "safe to edit", "allowed", "go no go"),
    "update": ("update", "change", "modify", "set", "fix", "restore", "reactivate", "reopen", "rename", "mark", "correct"),
    "notify": ("notify", "inform", "alert", "message", "let know", "heads up", "tell", "contact"),
    "delete": ("delete", "erase", "remove permanently", "wipe", "purge", "irreversible", "unrecoverable"),
    "read": ("read", "open", "show contents", "what does", "inside", "body", "written"),
    "inspect": ("inspect", "extract", "metadata", "properties", "attributes", "structured fields"),
    "export": ("export", "convert", "rendition", "downloadable", "pdf", "package"),
    "create": ("create", "add", "open a", "file", "build", "generate", "make", "produce", "new"),
    "archive": ("archive", "retain", "put away", "long term", "remove from active", "kept for audit"),
}

_EN_OBJECTS = {
    "user": ("user", "account", "profile", "person", "whoever", "sign in"),
    "document": ("document", "file", "text", "contents", "prose", "title"),
    "repository": ("repository", "repo", "codebase", "source project", "project", "code"),
    "issue": ("issue", "ticket", "work item", "task", "tracker", "work board"),
    "report": ("report", "analysis", "analytical deliverable", "digest"),
    "artifact": ("artifact", "deliverable", "vault entry", "retained item"),
}

_MULTILINGUAL_OPERATIONS = {
    "pt": {
        "search": ("encontre", "localize", "descubra"), "get": ("mostre", "consulte"),
        "validate": ("confirme", "verifique", "permitido"), "update": ("mude", "reative", "coloque", "corrija", "passe a chamar", "reabra"),
        "notify": ("avise", "informe"), "delete": ("elimine", "apague"), "read": ("abra", "conteudo", "texto"),
        "inspect": ("extraia", "metadados", "atributos"), "export": ("exporte", "versao pdf"),
        "create": ("crie", "gere", "abra uma nova"), "archive": ("arquive", "retire", "retire da area ativa"),
    },
    "es": {
        "search": ("encuentra", "localiza", "averigua"), "get": ("muestra", "consulta"),
        "validate": ("confirma", "verifica", "permitido"), "update": ("cambia", "reactiva", "pon", "corrige", "haz que", "se llame", "reabre"),
        "notify": ("avisa", "informa"), "delete": ("elimina", "borra"), "read": ("abre", "contenido", "texto"),
        "inspect": ("extrae", "metadatos", "atributos"), "export": ("exporta", "version pdf"),
        "create": ("crea", "genera", "abre una nueva"), "archive": ("archiva", "retira", "retira del area activa"),
    },
    "fr": {
        "search": ("trouvez", "localisez", "identifiez"), "get": ("affichez", "consultez"),
        "validate": ("confirmez", "verifiez", "autorisee"), "update": ("remplacez", "reactivez", "remettez", "corrigez", "nommez desormais", "rouvrez"),
        "notify": ("prevenez", "informez"), "delete": ("supprimez", "effacez"), "read": ("ouvrez", "contenu", "texte"),
        "inspect": ("extrayez", "metadonnees", "attributs"), "export": ("exportez", "version pdf"),
        "create": ("creez", "generez", "ouvrez une nouvelle"), "archive": ("archivez", "retirez", "retirez de l espace actif"),
    },
}

_MULTILINGUAL_OBJECTS = {
    "pt": {"user": ("utilizador", "conta", "quem", "u17"), "document": ("documento", "ficheiro", "d42"), "repository": ("repositorio", "projeto de codigo", "repo9"), "issue": ("incidencia", "tarefa", "issue 4"), "report": ("relatorio", "analise", "report 7"), "artifact": ("artefacto", "arquivo", "artifact old", "artifact d42 pdf")},
    "es": {"user": ("usuario", "cuenta", "quien", "u17"), "document": ("documento", "archivo", "d42"), "repository": ("repositorio", "proyecto de codigo", "repo9"), "issue": ("incidencia", "tarea", "issue 4"), "report": ("informe", "analisis", "report 7"), "artifact": ("artefacto", "archivo", "artifact old", "artifact d42 pdf")},
    "fr": {"user": ("utilisateur", "compte", "personne", "u17"), "document": ("document", "fichier", "d42"), "repository": ("depot", "projet de code", "repo9"), "issue": ("ticket", "tache", "issue 4"), "report": ("rapport", "analyse", "report 7"), "artifact": ("artefact", "archives", "artifact old", "artifact d42 pdf")},
}


def canonical_concept_map() -> CanonicalConceptMap:
    """Return project-authored English and multilingual concept mappings."""

    rows = []
    for canonical, surfaces in _EN_OPERATIONS.items():
        rows.extend(ConceptExpansion(surface, canonical, "operation", "en", "project_curated_en_v1") for surface in surfaces)
    for canonical, surfaces in _EN_OBJECTS.items():
        rows.extend(ConceptExpansion(surface, canonical, "object", "en", "project_domain_tools_v1") for surface in surfaces)
    for language, concepts in _MULTILINGUAL_OPERATIONS.items():
        for canonical, surfaces in concepts.items():
            rows.extend(ConceptExpansion(surface, canonical, "operation", language, "project_multilingual_v1") for surface in surfaces)
    for language, concepts in _MULTILINGUAL_OBJECTS.items():
        for canonical, surfaces in concepts.items():
            rows.extend(ConceptExpansion(surface, canonical, "object", language, "project_multilingual_v1") for surface in surfaces)
    return CanonicalConceptMap(rows)


def dictionary_sources_manifest() -> dict[str, object]:
    """Describe mapping provenance and considered third-party resources."""

    return {
        "schema_version": "1.0",
        "runtime_web_dependency": False,
        "active_sources": [
            {"id": "project_curated_en_v1", "kind": "general English operation concepts", "license": "repository license", "languages": ["en"]},
            {"id": "project_domain_tools_v1", "kind": "typed tool object concepts", "license": "repository license", "languages": ["en"]},
            {"id": "project_multilingual_v1", "kind": "bounded multilingual mappings", "license": "repository license", "languages": ["pt", "es", "fr"]},
            {"id": "tool_author_metadata", "kind": "operation, object, tags, schema inputs/outputs", "license": "catalog owner", "languages": ["en"]},
        ],
        "considered_not_embedded": [
            {"name": "Open English WordNet", "reason": "not needed for the bounded frozen catalog; avoids shipping a broad third-party lexical database"},
        ],
    }
