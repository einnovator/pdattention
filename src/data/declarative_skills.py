"""Declarative-skill catalog and semantic-hard requests for Paper 6.5 M9."""

from __future__ import annotations

from dataclasses import dataclass

from pra_hf.skill_records import SkillRecord


@dataclass(frozen=True)
class SkillQuery:
    """One held-out request with deterministic skill-following checks."""

    query_id: str
    split: str
    query: str
    target_skill: str
    family: str


@dataclass(frozen=True)
class _SkillSpec:
    name: str
    family: str
    description: str
    when_to_use: str
    validation_query: str
    test_query: str
    steps: tuple[str, str, str]
    constraint: str
    aliases: tuple[str, ...] = ()


_SPECS = (
    _SkillSpec("github_issue_triage", "engineering_intake", "Prioritize incoming repository issues using impact, evidence, and reproducibility.", "Use for a mixed queue of newly reported repository problems.", "Sort these new bug reports so engineering knows what to investigate first.", "Work through the incoming tickets and identify which reports deserve developer attention now.", ("Separate defects from questions and feature requests.", "Assess reproducibility and user impact.", "Assign priority and a concrete owner or next investigation."), "Do not close or downgrade a report without cited evidence.", ("issue intake review",)),
    _SkillSpec("github_pr_review", "engineering_review", "Review a proposed code change for correctness, risk, and maintainability.", "Use when a pull request or patch needs engineering review.", "Inspect this proposed patch and call out defects before it merges.", "Check the submitted code change for regressions, unsafe assumptions, and missing tests.", ("Identify the intended behavioral change.", "Trace correctness, compatibility, and security risks.", "Report actionable findings ordered by severity."), "Do not approve a change while a blocking finding remains.", ("patch review", "merge review")),
    _SkillSpec("incident_triage", "operations", "Classify an active service incident and establish an immediate response path.", "Use when production behavior is degraded or an alert may require escalation.", "The service is failing intermittently; establish severity and the first response actions.", "Help the on-call engineer decide whether this alert is an incident and what to stabilize first.", ("Confirm observable customer and system impact.", "Assign severity from evidence and current blast radius.", "Name containment, owner, and next diagnostic."), "Do not claim recovery until a health signal confirms it.", ("on-call intake",)),
    _SkillSpec("postmortem_writing", "operations", "Turn an incident timeline into a blameless causal postmortem.", "Use after service recovery when causes and corrective actions must be documented.", "Turn this outage timeline into a blameless account with corrective actions.", "Document what led to the failure, how it was detected, and how recurrence will be prevented.", ("Reconstruct the evidence-backed timeline.", "Separate contributing conditions from the trigger.", "Assign measurable corrective actions and owners."), "Do not attribute intent or personal blame.", ("incident retrospective",)),
    _SkillSpec("document_review", "document_analysis", "Review a document for internal consistency, completeness, and audience fit.", "Use when prose or a specification needs editorial and structural review.", "Check this specification for contradictions, missing sections, and unclear wording.", "Read the draft as an editor and identify where its logic or presentation breaks down.", ("Identify the document's audience and purpose.", "Check claims, structure, and terminology for consistency.", "Propose bounded edits with reasons."), "Do not invent facts that the source does not support.", ("editorial review",)),
    _SkillSpec("research_summary", "document_analysis", "Extract claims, evidence, limitations, and open questions from research material.", "Use when papers or technical reports must be compared for reading or research decisions.", "Pull out the main claims and unresolved questions from these papers.", "Condense this research material into evidence, limitations, and the questions worth pursuing next.", ("Separate stated claims from measured evidence.", "Record limitations and conflicting results.", "Synthesize open questions without erasing source differences."), "Do not present inference as a reported result.", ("paper synthesis", "evidence summary")),
    _SkillSpec("meeting_summary", "communication", "Convert meeting notes into decisions, owners, and follow-up actions.", "Use after a meeting when participants need an operational record.", "Turn these rough meeting notes into decisions and assigned follow-ups.", "Capture what the group agreed, who owns each action, and what remains unresolved.", ("Separate discussion from decisions.", "Extract actions, owners, and dates when stated.", "List unresolved questions explicitly."), "Do not assign an owner that the notes do not identify.", ("meeting minutes",)),
    _SkillSpec("customer_support_reply", "communication", "Draft an accurate, empathetic support response grounded in known product facts.", "Use for a routine customer problem that needs explanation or troubleshooting.", "Respond to this customer who cannot complete setup and needs clear next steps.", "Write a helpful answer to the user, acknowledge the problem, and guide them through a safe fix.", ("Restate the customer's observed problem.", "Offer ordered troubleshooting grounded in supplied facts.", "State escalation criteria and the next contact path."), "Do not promise a fix date or action that is not confirmed.", ("support response",)),
    _SkillSpec("customer_escalation", "communication", "Prepare a high-risk customer escalation with impact, evidence, and ownership.", "Use when a customer problem is severe, repeated, blocked, or contract-sensitive.", "Prepare this repeated enterprise failure for urgent escalation with the right evidence.", "Package the account impact and failed remedies so leadership and engineering can take ownership.", ("Quantify user, business, and contractual impact.", "Summarize attempted remedies and remaining uncertainty.", "Name the requested decision, owner, and response deadline."), "Do not minimize unresolved impact or fabricate commitments.", ("account escalation",)),
    _SkillSpec("bug_reproduction", "engineering_diagnosis", "Convert a defect report into a minimal repeatable reproduction protocol.", "Use when a reported software failure is too ambiguous for diagnosis.", "Turn this vague failure report into steps another engineer can reproduce.", "Reduce the reported defect to a controlled case with inputs, environment, and expected versus actual behavior.", ("Record environment and preconditions.", "Minimize the action sequence and input data.", "State expected, actual, and repeatability observations."), "Do not label a hypothesis as the root cause.", ("repro case",)),
    _SkillSpec("performance_diagnosis", "engineering_diagnosis", "Diagnose a performance regression from measurements and controlled comparisons.", "Use when latency, throughput, or resource use has degraded.", "Investigate why this endpoint became slower after the release.", "Use the timing and resource measurements to narrow the source of the regression.", ("Define the baseline and regression interval.", "Partition compute, I/O, queueing, and allocation costs.", "Propose the smallest discriminating measurement next."), "Do not infer causality from one correlated metric.", ("latency investigation",)),
    _SkillSpec("data_quality_review", "data_governance", "Assess a dataset for validity, completeness, consistency, and leakage.", "Use before analysis, training, or release of a dataset.", "Check this table for missing values, inconsistent fields, and leakage risks.", "Review whether this dataset is trustworthy enough for modeling and identify the repairs it needs.", ("Profile schema, missingness, ranges, and duplicates.", "Test cross-field consistency and label leakage.", "Prioritize repairs and define acceptance checks."), "Do not silently discard anomalous records.", ("dataset audit",)),
    _SkillSpec("privacy_assessment", "data_governance", "Evaluate data handling for privacy exposure, minimization, and retention risk.", "Use when a workflow collects, stores, shares, or derives personal information.", "Assess whether this proposed data flow collects more personal information than necessary.", "Review the workflow for privacy exposure, retention, and inappropriate secondary use.", ("Map personal data, purpose, storage, and recipients.", "Check minimization, consent, retention, and deletion controls.", "State residual risks and required approvals."), "Do not treat de-identification as guaranteed anonymity.", ("privacy review",)),
    _SkillSpec("security_review", "assurance", "Review a design for threat boundaries, abuse paths, and control gaps.", "Use before deploying a security-sensitive service or integration.", "Threat-model this service change and identify the controls missing before launch.", "Examine how an attacker could misuse the proposed integration and what should block them.", ("Identify assets, actors, trust boundaries, and entry points.", "Enumerate plausible abuse paths and consequences.", "Map mitigations, detection, and residual risk."), "Do not claim a design is secure solely because authentication exists.", ("threat review",)),
    _SkillSpec("dependency_audit", "assurance", "Review software dependencies for provenance, vulnerability, license, and maintenance risk.", "Use before adopting or upgrading third-party packages.", "Assess these libraries for security, license, and maintenance concerns.", "Decide which third-party packages are safe to adopt and which require mitigation or replacement.", ("Verify source, version, and transitive dependency inventory.", "Check vulnerability, license, and maintainer signals.", "Recommend pin, mitigate, replace, or reject with evidence."), "Do not infer safety from download count alone.", ("package due diligence",)),
    _SkillSpec("release_checklist", "delivery", "Verify release readiness across tests, artifacts, migration, rollback, and communication.", "Use immediately before publishing a software release.", "Check whether this build is ready to ship and identify every blocking item.", "Walk through the evidence needed before we publish this version.", ("Verify tests, artifacts, versions, and provenance.", "Check migration, compatibility, monitoring, and rollback.", "Classify blockers and record the release decision."), "Do not mark ready while an unresolved blocker exists.", ("ship readiness",)),
    _SkillSpec("migration_planning", "delivery", "Plan a staged system or data migration with compatibility and rollback controls.", "Use when moving traffic, data, APIs, or infrastructure between implementations.", "Plan how to move this service without breaking clients or losing data.", "Create a staged cutover path with verification and a usable rollback.", ("Inventory dependencies, compatibility, and data invariants.", "Define stages, validation gates, and ownership.", "Specify rollback triggers and reconciliation."), "Do not schedule irreversible cutover before rollback is tested.", ("cutover plan",)),
    _SkillSpec("api_design_review", "design_review", "Review an API contract for consistency, evolvability, failure semantics, and usability.", "Use when introducing or changing a public or internal interface.", "Review this endpoint contract for compatibility and confusing behavior.", "Check whether clients can use and evolve with this interface safely.", ("Model resources, operations, and caller workflows.", "Check naming, validation, errors, idempotency, and pagination.", "Identify compatibility risks and concrete contract changes."), "Do not recommend a breaking change without a migration path.", ("interface review",)),
    _SkillSpec("accessibility_review", "design_review", "Review an interface for keyboard, screen-reader, contrast, and cognitive accessibility.", "Use when evaluating a user-facing workflow or component.", "Audit this form for barriers faced by keyboard and screen-reader users.", "Check whether people with different access needs can complete this interface reliably.", ("Trace keyboard order, focus, labels, and announcements.", "Check visual contrast, scaling, motion, and error recovery.", "Report barriers by task impact with testable fixes."), "Do not equate automated checks with complete accessibility validation.", ("a11y audit",)),
    _SkillSpec("experiment_analysis", "research_method", "Analyze an experiment using estimands, uncertainty, controls, and failure modes.", "Use after collecting controlled experimental measurements.", "Interpret these trial results and separate measured effects from noise.", "Analyze whether the intervention helped, how certain that conclusion is, and what the controls reveal.", ("Restate the hypothesis, estimand, cohort, and exclusions.", "Quantify effects and uncertainty against controls.", "Inspect heterogeneity, failure modes, and decision implications."), "Do not treat statistical significance as practical importance.", ("experiment interpretation",)),
    _SkillSpec("model_eval_review", "research_method", "Review a model evaluation for validity, leakage, calibration, and behavioral coverage.", "Use when deciding whether reported model metrics support deployment or comparison claims.", "Audit this model benchmark for leakage, weak controls, and misleading aggregate scores.", "Determine whether the evaluation really supports the claimed model improvement.", ("Check task construction, splits, contamination, and baselines.", "Review metrics, uncertainty, calibration, and subgroup behavior.", "Identify unsupported claims and the next decisive evaluation."), "Do not collapse safety and quality into one aggregate score.", ("evaluation audit",)),
    _SkillSpec("prompt_eval_design", "research_method", "Design a reproducible prompt evaluation with matched inputs and scored outcomes.", "Use before comparing prompts, instructions, or agent policies.", "Set up a fair test of these two system prompts across realistic requests.", "Design an evaluation that can tell whether the instruction change actually improves behavior.", ("Define target behaviors, failure classes, and held-out inputs.", "Freeze model settings and matched comparison conditions.", "Specify deterministic scoring plus blinded review where needed."), "Do not tune on the final test set.", ("instruction benchmark design",)),
    _SkillSpec("literature_screening", "research_workflow", "Screen a literature set against explicit inclusion and exclusion criteria.", "Use during evidence review before full-text synthesis.", "Filter these abstracts to the studies that answer our question and record why others are excluded.", "Apply the review criteria consistently and produce a defensible shortlist for full reading.", ("Translate the research question into screening criteria.", "Judge each record using only available evidence.", "Record include, exclude, or uncertain with a reason."), "Do not infer unavailable methods from an abstract.", ("paper screening",)),
    _SkillSpec("onboarding_plan", "people_process", "Create a role-specific onboarding sequence with access, learning, and feedback checkpoints.", "Use when a new team member needs a bounded ramp-up plan.", "Build a first-month plan for the new engineer with access, learning, and review milestones.", "Organize how this teammate should gain context and contribute safely during the first weeks.", ("Identify role outcomes, prerequisites, and access dependencies.", "Sequence learning, shadowing, and bounded contributions.", "Set feedback checkpoints and observable completion criteria."), "Do not grant production authority before required access review.", ("ramp-up plan",)),
    _SkillSpec("decision_record", "people_process", "Document an architectural decision with context, options, rationale, and consequences.", "Use when a durable technical choice needs traceable reasoning.", "Capture why the team chose this architecture and what alternatives were rejected.", "Turn the design discussion into a durable note future engineers can understand and revisit.", ("State context, constraints, and decision drivers.", "Compare considered options and evidence.", "Record the decision, consequences, and revisit triggers."), "Do not rewrite uncertainty as unanimous certainty.", ("architecture decision note",)),
)


def declarative_skill_catalog() -> tuple[SkillRecord, ...]:
    """Return 24 non-executable skills with realistic full instruction bodies."""

    records = []
    for spec in _SPECS:
        body = (
            f"Apply the {spec.name} procedure to the user's supplied material. "
            "Keep observations separate from assumptions and make the response useful to the named decision maker. "
            f"The procedure is: (1) {spec.steps[0]} (2) {spec.steps[1]} (3) {spec.steps[2]} "
            f"Mandatory constraint: {spec.constraint} "
            f"Begin the final response with `SKILL_APPLIED: {spec.name}`. Then use the headings "
            "`Decision`, `Evidence`, and `Next action`. State missing evidence rather than fabricating it. "
            "This skill provides instructions only: never claim that an external service, file, account, or system was changed."
        )
        records.append(SkillRecord(
            name=spec.name,
            description=spec.description,
            when_to_use=spec.when_to_use,
            instructions=body,
            namespace="paper6_5",
            tenant_id="paper6_5",
            aliases=spec.aliases,
            manual_tags=frozenset((spec.family, *spec.aliases)),
            constraints=(spec.constraint, "Never claim an external action was executed."),
            ordered_steps=spec.steps,
            examples=(
                f"SKILL_APPLIED: {spec.name}\nDecision: ...\nEvidence: ...\nNext action: ...",
            ),
            dependencies=("user_supplied_material",),
            references=("paper6_5_declarative_skill_contract",),
            metadata={"family": spec.family, "benchmark": "paper6_5_m9"},
        ))
    return tuple(records)


def skill_semantic_hard_queries() -> tuple[SkillQuery, ...]:
    """Return paired validation/test paraphrases without canonical skill names."""

    rows = []
    for spec in _SPECS:
        rows.extend((
            SkillQuery(f"{spec.name}-validation", "validation", spec.validation_query, spec.name, spec.family),
            SkillQuery(f"{spec.name}-test", "test", spec.test_query, spec.name, spec.family),
        ))
    return tuple(rows)
