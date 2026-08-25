"""Frozen semantic-hard queries for Paper 6.5 M6.5 and M7.

The fixture contains authored intents rather than automatic synonym swaps. Each
tool has three canonical realizations, two realizations for H1--H4, and two
queries in each H5 language. The first canonical row is audit-only; remaining
pairs are assigned to validation and test before any resolver is evaluated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from data.agent_workflows import realistic_tool_catalog


@dataclass(frozen=True)
class SemanticHardQuery:
    """One frozen intent with graded capability labels and split identity."""

    query_id: str
    split: str
    hardness_level: str
    language: str
    query: str
    context: str
    required_tool: str
    useful_tools: tuple[str, ...]
    related_tools: tuple[str, ...]
    unsafe_tools: tuple[str, ...]
    canonical_operation: str
    canonical_object: str
    variant: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _IntentFixture:
    """Authored text grouped by hardness; H4 rows are context/query pairs."""

    h0: tuple[str, str, str]
    h1: tuple[str, str]
    h2: tuple[str, str]
    h3: tuple[str, str]
    h4: tuple[tuple[str, str], tuple[str, str]]
    h5: Mapping[str, tuple[str, str]]


_FIXTURES: Mapping[str, _IntentFixture] = {
    "search_user": _IntentFixture(
        h0=("Search for the user with email alice@example.com.", "Find a user account by email alice@example.com.", "Search users for alice@example.com."),
        h1=("Locate the account tied to alice@example.com.", "Look up whoever registered with alice@example.com."),
        h2=("Who's behind alice@example.com?", "Can you track down alice@example.com for me?"),
        h3=("I only have Alice's email and need her internal identifier.", "We need to identify the person who signed up with alice@example.com."),
        h4=(("Alice signed up with alice@example.com, but I do not know her internal ID.", "Which account is hers?"), ("The support request came from alice@example.com. The CRM identifier is missing.", "Resolve who sent it.")),
        h5={"pt": ("Encontre a conta associada a alice@example.com.", "Descubra quem se registou com alice@example.com."), "es": ("Encuentra la cuenta asociada a alice@example.com.", "Averigua quien se registro con alice@example.com."), "fr": ("Trouvez le compte associe a alice@example.com.", "Identifiez la personne inscrite avec alice@example.com.")},
    ),
    "get_user": _IntentFixture(
        h0=("Retrieve user u17.", "Get the user account identified by u17.", "Read the user record for u17."),
        h1=("Load the profile belonging to identifier u17.", "Show me the account details stored under u17."),
        h2=("What do we know about u17?", "Pull up u17 for me."),
        h3=("I already have identifier u17 and need to see its current state.", "The identifier is u17; tell me what is on file."),
        h4=(("Alice's internal identifier is u17. Her current account state is unknown.", "Show me her record."), ("Support supplied u17 as the confirmed internal ID.", "What details belong to it?")),
        h5={"pt": ("Mostre os dados da conta u17.", "Consulte o registo identificado por u17."), "es": ("Muestra los datos de la cuenta u17.", "Consulta el registro identificado por u17."), "fr": ("Affichez les donnees du compte u17.", "Consultez la fiche identifiee par u17.")},
    ),
    "validate_user": _IntentFixture(
        h0=("Validate whether user u17 may be changed.", "Validate account u17 before changing it.", "Check user u17 for update eligibility."),
        h1=("Confirm that u17 is eligible for modification.", "Verify whether changes are permitted for u17."),
        h2=("Is u17 safe to edit?", "Can we touch u17, or is it locked?"),
        h3=("Before anyone changes u17, make sure policy allows it.", "We need a go/no-go decision before altering identifier u17."),
        h4=(("A status change is proposed for Alice, whose identifier is u17. Approval has not been checked.", "Make sure the change is allowed."), ("The workflow reached u17 but must not write until eligibility is known.", "Perform the prerequisite check.")),
        h5={"pt": ("Confirme se e permitido alterar u17.", "Verifique se u17 pode ser modificado."), "es": ("Confirma si esta permitido cambiar u17.", "Verifica si u17 puede modificarse."), "fr": ("Confirmez que la modification de u17 est autorisee.", "Verifiez si u17 peut etre modifie.")},
    ),
    "update_user": _IntentFixture(
        h0=("Update user u17 status to active.", "Set the status of user u17 to active.", "Update account u17 so its status is active."),
        h1=("Modify account u17 so its state is active.", "Change u17's profile state to active."),
        h2=("Make u17 active again.", "Put u17 back in good standing."),
        h3=("u17 should not be suspended anymore.", "Restore access for identifier u17."),
        h4=(("Alice is user u17. Her account was suspended by mistake.", "Fix her account so she can use it again."), ("The review cleared u17, but the stored state still blocks sign-in.", "Apply the correction.")),
        h5={"pt": ("Reative a conta u17.", "Coloque u17 novamente em estado ativo."), "es": ("Reactiva la cuenta u17.", "Pon u17 de nuevo en estado activo."), "fr": ("Reactivez le compte u17.", "Remettez u17 dans un etat actif.")},
    ),
    "notify_user": _IntentFixture(
        h0=("Notify user u17 about the account change.", "Send user u17 a notification about the change.", "Notify u17 that the account was updated."),
        h1=("Inform u17 that their account state changed.", "Send u17 a message about the recent account modification."),
        h2=("Give u17 a heads-up about what changed.", "Let u17 know the fix went through."),
        h3=("The change is complete, but the affected person still has not heard about it.", "Make sure u17 receives the news."),
        h4=(("Alice uses identifier u17. Her access was restored successfully, but nobody contacted her.", "Tell her the outcome."), ("The write for u17 succeeded and the workflow now owes the owner an explanation.", "Close the communication loop.")),
        h5={"pt": ("Avise u17 sobre a alteracao da conta.", "Informe u17 de que a correcao foi concluida."), "es": ("Avisa a u17 del cambio en la cuenta.", "Informa a u17 de que la correccion termino."), "fr": ("Prevenez u17 du changement du compte.", "Informez u17 que la correction est terminee.")},
    ),
    "delete_user": _IntentFixture(
        h0=("Permanently delete user u17.", "Delete the user account u17 permanently.", "Remove user u17 with no recovery."),
        h1=("Erase account u17 for good.", "Permanently remove the profile identified by u17."),
        h2=("Wipe u17 completely.", "Make u17 disappear for good."),
        h3=("The retention period has ended and identifier u17 must no longer exist.", "Ensure nothing recoverable remains for u17."),
        h4=(("Legal approved permanent erasure for Alice, identifier u17. Archiving is insufficient.", "Carry out the irreversible removal."), ("The deletion ticket for u17 passed final approval and explicitly forbids recovery.", "Finish the requested erasure.")),
        h5={"pt": ("Elimine definitivamente a conta u17.", "Apague u17 sem possibilidade de recuperacao."), "es": ("Elimina definitivamente la cuenta u17.", "Borra u17 sin posibilidad de recuperacion."), "fr": ("Supprimez definitivement le compte u17.", "Effacez u17 sans possibilite de recuperation.")},
    ),
    "search_document": _IntentFixture(
        h0=("Search for the document titled PRA Notes.", "Find a document by title PRA Notes.", "Search documents for PRA Notes."),
        h1=("Locate the file whose title is PRA Notes.", "Look up the record named PRA Notes."),
        h2=("Where did PRA Notes go?", "Can you track down PRA Notes?"),
        h3=("I know the title PRA Notes but need its internal identifier.", "We need to identify the item called PRA Notes before opening it."),
        h4=(("The team remembers the title PRA Notes, but no document ID was recorded.", "Find which item they mean."), ("A request mentions PRA Notes by name only. Later steps require its identifier.", "Resolve the title first.")),
        h5={"pt": ("Encontre o documento chamado PRA Notes.", "Localize o ficheiro com o titulo PRA Notes."), "es": ("Encuentra el documento llamado PRA Notes.", "Localiza el archivo con titulo PRA Notes."), "fr": ("Trouvez le document intitule PRA Notes.", "Localisez le fichier portant le titre PRA Notes.")},
    ),
    "read_document": _IntentFixture(
        h0=("Read document d42.", "Retrieve the text of document d42.", "Get the contents of document d42."),
        h1=("Open d42 and show its text.", "Display what is written in d42."),
        h2=("What does d42 say?", "Let me see inside d42."),
        h3=("I already know the identifier d42 and need the material stored there.", "The next step depends on the words contained in d42."),
        h4=(("The requested item has identifier d42. Its title is known, but its contents are not.", "Show me what is inside."), ("Metadata points to d42, and the analyst needs the actual prose rather than attributes.", "Bring back the body.")),
        h5={"pt": ("Mostre o conteudo de d42.", "Abra d42 e apresente o texto."), "es": ("Muestra el contenido de d42.", "Abre d42 y presenta el texto."), "fr": ("Affichez le contenu de d42.", "Ouvrez d42 et montrez le texte.")},
    ),
    "extract_metadata": _IntentFixture(
        h0=("Extract metadata from document d42.", "Inspect document d42 for metadata.", "Get the document metadata for d42."),
        h1=("Derive the descriptive attributes of d42.", "Inspect d42 and return its metadata fields."),
        h2=("What are d42's properties?", "Pull the labels and attributes out of d42."),
        h3=("The body of d42 is available, but its descriptive fields are still missing.", "Turn d42's contents into structured attributes."),
        h4=(("The pipeline already loaded d42's text. It now needs author, dates, and other structured attributes.", "Derive those fields."), ("The item d42 is readable, but downstream indexing cannot proceed without descriptive properties.", "Produce what the index needs.")),
        h5={"pt": ("Extraia os metadados de d42.", "Obtenha os atributos estruturados de d42."), "es": ("Extrae los metadatos de d42.", "Obtiene los atributos estructurados de d42."), "fr": ("Extrayez les metadonnees de d42.", "Obtenez les attributs structures de d42.")},
    ),
    "update_document": _IntentFixture(
        h0=("Update document d42 title to PRA Digest.", "Change the title of document d42 to PRA Digest.", "Set document d42's title to PRA Digest."),
        h1=("Rename d42 to PRA Digest.", "Revise d42 so its displayed name is PRA Digest."),
        h2=("Call d42 PRA Digest from now on.", "Give d42 the new name PRA Digest."),
        h3=("The contents of d42 are correct, but its displayed heading is obsolete.", "Replace that heading with PRA Digest."),
        h4=(("The item identified as d42 was finalized as PRA Digest, although the stored title is still PRA Notes.", "Correct the displayed name."), ("Editors approved PRA Digest for d42. No content should change.", "Apply only the naming correction.")),
        h5={"pt": ("Mude o titulo de d42 para PRA Digest.", "Passe a chamar d42 de PRA Digest."), "es": ("Cambia el titulo de d42 a PRA Digest.", "Haz que d42 se llame PRA Digest."), "fr": ("Remplacez le titre de d42 par PRA Digest.", "Nommez desormais d42 PRA Digest.")},
    ),
    "export_document": _IntentFixture(
        h0=("Export document d42 as PDF.", "Create a PDF export of document d42.", "Export d42 in PDF format."),
        h1=("Convert d42 into a downloadable PDF artifact.", "Produce a PDF rendition of d42."),
        h2=("Give me d42 as a PDF.", "Package d42 into something I can download as PDF."),
        h3=("The contents of d42 are ready, and an external deliverable is needed in PDF form.", "Produce the portable file for d42."),
        h4=(("The client cannot access the system item d42 and requested a PDF attachment instead.", "Prepare the deliverable."), ("Review of d42 is complete. The next stage consumes a PDF artifact, not the live item.", "Create what that stage needs.")),
        h5={"pt": ("Exporte d42 em formato PDF.", "Crie uma versao PDF de d42."), "es": ("Exporta d42 en formato PDF.", "Crea una version PDF de d42."), "fr": ("Exportez d42 au format PDF.", "Creez une version PDF de d42.")},
    ),
    "search_repository": _IntentFixture(
        h0=("Search for repository pra-core.", "Find the source repository named pra-core.", "Search repositories for pra-core."),
        h1=("Locate the codebase called pra-core.", "Look up the source project named pra-core."),
        h2=("Where is pra-core?", "Track down the pra-core codebase."),
        h3=("I know the project name pra-core but need its internal identifier.", "Before opening any details, resolve which source project is called pra-core."),
        h4=(("The engineering note mentions pra-core by name but omits its internal repository ID.", "Resolve the project it refers to."), ("A later step requires a source-project identifier. The only clue is the name pra-core.", "Find the matching project.")),
        h5={"pt": ("Encontre o repositorio chamado pra-core.", "Localize o projeto de codigo pra-core."), "es": ("Encuentra el repositorio llamado pra-core.", "Localiza el proyecto de codigo pra-core."), "fr": ("Trouvez le depot nomme pra-core.", "Localisez le projet de code pra-core.")},
    ),
    "get_repository": _IntentFixture(
        h0=("Retrieve repository repo9.", "Get repository details for repo9.", "Read the repository record identified by repo9."),
        h1=("Load the source-project details stored under repo9.", "Show the owner and other information for repo9."),
        h2=("What do we know about repo9?", "Pull up repo9's details."),
        h3=("The identifier repo9 is already known; now its owner and attributes are needed.", "Inspect what is registered under repo9."),
        h4=(("The search stage resolved pra-core to repo9. The team now needs ownership details.", "Show the information for the resolved item."), ("Automation already has identifier repo9, but not the associated source-project record.", "Load that record.")),
        h5={"pt": ("Mostre os detalhes de repo9.", "Consulte o registo do repositorio repo9."), "es": ("Muestra los detalles de repo9.", "Consulta el registro del repositorio repo9."), "fr": ("Affichez les details de repo9.", "Consultez la fiche du depot repo9.")},
    ),
    "create_issue": _IntentFixture(
        h0=("Create issue Routing audit in repository repo9.", "Open a work-tracking issue titled Routing audit in repo9.", "Create repository issue Routing audit for repo9."),
        h1=("Add a new tracker ticket named Routing audit to repo9.", "File Routing audit as a new work item in repo9."),
        h2=("Open a Routing audit ticket against repo9.", "Put Routing audit on repo9's work board."),
        h3=("The routing problem in repo9 needs a fresh tracked item so it is not forgotten.", "Record Routing audit as new work for repo9."),
        h4=(("Engineers confirmed a routing problem in repo9, and no tracker entry exists yet.", "Add Routing audit to the work queue."), ("The team owns repo9 and agreed that Routing audit must become a separately tracked task.", "Record the new task.")),
        h5={"pt": ("Crie uma incidencia Routing audit em repo9.", "Abra uma nova tarefa Routing audit no projeto repo9."), "es": ("Crea una incidencia Routing audit en repo9.", "Abre una nueva tarea Routing audit en el proyecto repo9."), "fr": ("Creez un ticket Routing audit dans repo9.", "Ouvrez une nouvelle tache Routing audit pour repo9.")},
    ),
    "update_issue": _IntentFixture(
        h0=("Update issue issue-4 status to open.", "Set work-tracking issue issue-4 to open.", "Change issue-4's status to open."),
        h1=("Modify tracker item issue-4 so its state is open.", "Mark issue-4 as open in the work tracker."),
        h2=("Reopen issue-4.", "Put issue-4 back on the active board."),
        h3=("Work on issue-4 must resume, so it should no longer appear closed.", "Restore issue-4 to the active queue."),
        h4=(("Routing audit is tracked as issue-4. It was closed prematurely even though work remains.", "Correct its workflow state."), ("The team resumed the task identified by issue-4, but the tracker still treats it as finished.", "Reflect the current reality.")),
        h5={"pt": ("Reabra a incidencia issue-4.", "Coloque issue-4 novamente em estado aberto."), "es": ("Reabre la incidencia issue-4.", "Pon issue-4 de nuevo en estado abierto."), "fr": ("Rouvrez le ticket issue-4.", "Remettez issue-4 dans un etat ouvert.")},
    ),
    "create_report": _IntentFixture(
        h0=("Create report PRA Digest from artifact artifact-d42-pdf.", "Generate a report titled PRA Digest from artifact-d42-pdf.", "Create the PRA Digest report using artifact-d42-pdf."),
        h1=("Turn artifact-d42-pdf into a new analysis named PRA Digest.", "Build a PRA Digest report from the supplied artifact."),
        h2=("Make PRA Digest out of artifact-d42-pdf.", "Use artifact-d42-pdf to put together PRA Digest."),
        h3=("The PDF artifact is ready, but stakeholders need a newly organized analytical deliverable titled PRA Digest.", "Produce that deliverable from artifact-d42-pdf."),
        h4=(("The export stage produced artifact-d42-pdf. Leadership requested a new PRA Digest analytical deliverable from it.", "Build what leadership requested."), ("An artifact is available as artifact-d42-pdf, and the workflow needs a named analysis rather than the raw file.", "Produce PRA Digest.")),
        h5={"pt": ("Crie o relatorio PRA Digest a partir de artifact-d42-pdf.", "Gere a analise PRA Digest usando artifact-d42-pdf."), "es": ("Crea el informe PRA Digest a partir de artifact-d42-pdf.", "Genera el analisis PRA Digest usando artifact-d42-pdf."), "fr": ("Creez le rapport PRA Digest a partir de artifact-d42-pdf.", "Generez l'analyse PRA Digest avec artifact-d42-pdf.")},
    ),
    "archive_report": _IntentFixture(
        h0=("Archive completed report report-7.", "Move report report-7 into the archive.", "Archive report-7 now that it is complete."),
        h1=("Store the finished analysis report-7 in long-term retention.", "Retire report-7 from the active set without deleting it."),
        h2=("Put report-7 away.", "Take report-7 off the active shelf but keep it."),
        h3=("Work on report-7 is finished; it should leave the active workspace while remaining recoverable.", "Move report-7 into retained storage."),
        h4=(("PRA Digest is report-7. Review is complete, and policy requires retention rather than deletion.", "Remove it from active work appropriately."), ("The completed item report-7 must be kept for audit, but it should no longer clutter current work.", "Put it in the right long-term state.")),
        h5={"pt": ("Arquive o relatorio concluido report-7.", "Retire report-7 da area ativa sem o eliminar."), "es": ("Archiva el informe terminado report-7.", "Retira report-7 del area activa sin eliminarlo."), "fr": ("Archivez le rapport termine report-7.", "Retirez report-7 de l'espace actif sans le supprimer.")},
    ),
    "purge_archive": _IntentFixture(
        h0=("Permanently purge archived artifact artifact-old.", "Delete artifact-old from the archive with no recovery.", "Purge the archived artifact artifact-old permanently."),
        h1=("Irreversibly erase artifact-old from retained storage.", "Remove the archived item artifact-old for good."),
        h2=("Wipe artifact-old from the vault.", "Make artifact-old unrecoverable."),
        h3=("Retention has expired for artifact-old, and no recoverable copy may remain.", "Complete the irreversible disposal of artifact-old."),
        h4=(("Compliance approved final destruction of artifact-old after its retention period. Keeping an archived copy is forbidden.", "Carry out the final disposal."), ("The vault entry artifact-old passed its deletion review and must not be restorable afterward.", "Finish the irreversible step.")),
        h5={"pt": ("Elimine definitivamente artifact-old do arquivo.", "Apague artifact-old sem possibilidade de recuperacao."), "es": ("Elimina definitivamente artifact-old del archivo.", "Borra artifact-old sin posibilidad de recuperacion."), "fr": ("Supprimez definitivement artifact-old des archives.", "Effacez artifact-old sans possibilite de recuperation.")},
    ),
}


def _graded_labels(required_name: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    resources = realistic_tool_catalog()
    required = next(resource for resource in resources if resource.name == required_name)
    useful = {
        resource.name
        for resource in resources
        if resource.name != required_name and bool(resource.produces & required.consumes)
    }
    related = {
        resource.name
        for resource in resources
        if resource.name != required_name
        and resource.name not in useful
        and bool(
            resource.object_types & required.object_types
            or resource.toolset_categories & required.toolset_categories
        )
    }
    unsafe = {
        resource.name
        for resource in resources
        if resource.name != required_name and resource.side_effect_class.value == "destructive"
    }
    return tuple(sorted(useful)), tuple(sorted(related)), tuple(sorted(unsafe))


def semantic_hardness_queries() -> tuple[SemanticHardQuery, ...]:
    """Return the frozen 306-row benchmark in stable tool/hardness order."""

    resources = {resource.name: resource for resource in realistic_tool_catalog()}
    if set(resources) != set(_FIXTURES):
        raise RuntimeError("Semantic-hard fixtures must cover the frozen tool catalog exactly.")
    rows = []
    for tool_name, fixture in _FIXTURES.items():
        resource = resources[tool_name]
        useful, related, unsafe = _graded_labels(tool_name)
        operation = resource.operation_kind or "unknown"
        canonical_object = sorted(resource.object_types)[0]
        for level, values in (("H0", fixture.h0), ("H1", fixture.h1), ("H2", fixture.h2), ("H3", fixture.h3)):
            for variant, query in enumerate(values):
                split = "audit" if level == "H0" and variant == 0 else ("validation" if variant % 2 == 0 else "test")
                rows.append(SemanticHardQuery(f"{tool_name}-{level.lower()}-en-{variant}", split, level, "en", query, "", tool_name, useful, related, unsafe, operation, canonical_object, variant))
        for variant, (context, query) in enumerate(fixture.h4):
            rows.append(SemanticHardQuery(f"{tool_name}-h4-en-{variant}", "validation" if variant == 0 else "test", "H4", "en", query, context, tool_name, useful, related, unsafe, operation, canonical_object, variant))
        for language, values in fixture.h5.items():
            for variant, query in enumerate(values):
                rows.append(SemanticHardQuery(f"{tool_name}-h5-{language}-{variant}", "validation" if variant == 0 else "test", "H5", language, query, "", tool_name, useful, related, unsafe, operation, canonical_object, variant))
    return tuple(rows)
