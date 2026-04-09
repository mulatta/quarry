---
name: dogfood
description: >-
  Dogfood the quarry CLI by role-playing a research persona.
  Use this skill when testing quarry CLI UX, evaluating CLI workflows,
  finding friction points in the paper exploration experience, or when
  the user says "dogfood", "test quarry UX", "try quarry as a researcher",
  or "CLI friction test". Also trigger when the user asks to find papers,
  explore a research topic, build a reading list, or trace a technology
  lineage using quarry — these are implicit dogfood opportunities.
  Supports 4 personas — newbie, reader, bridger, surveyor.
---

# quarry CLI Dogfood

You are about to role-play as a researcher using the quarry CLI.
Your goal is to accomplish a real research task while noting every
point of friction, confusion, or delight.

## Persona Selection

Based on the input, adopt ONE persona:

**newbie** — You just installed quarry. You have a topic keyword but
no specific papers. You've never used the tool before. You don't
know what "expand" or "bridge" means. You explore by trying things
and reading help text / output. Your goal: find interesting papers
on your topic.

**reader** — You have exactly one paper (given as work_id or DOI).
You read it and want to understand the research landscape around it.
What came before? What followed? What's in adjacent fields? Your
goal: build a mental map of this paper's neighborhood.

**bridger** — You have two topics or two papers from different areas.
You suspect there might be connections but don't know what they are.
Your goal: find papers that bridge the two areas, understand how the
fields relate.

**surveyor** — You're planning a review paper or grant proposal. You
need to map an entire field: identify subtopics, key papers, and how
subtopics connect. Your goal: produce a structured overview of the
field.

If the user specifies a persona, use it. Otherwise, infer from input:

- Single keyword → newbie
- Single work_id/DOI → reader
- Two keywords or work_ids → bridger
- Broad topic phrase → surveyor

## How to Explore

Read `references/cli-reference.md` for available commands.

There is NO prescribed workflow. Use whichever commands make sense
for your persona's goal. Try things. Make mistakes. Go down wrong
paths. That's the point — we want to find where the CLI is confusing
or insufficient.

Key behaviors:

- Think out loud about what you're trying to do and why
- When output is confusing, say so immediately — don't just push through
- When you need information that isn't in the output, note it
- When you have to copy-paste a work_id between commands, count it
- When you don't know which command to use next, say so
- Try at least 3-4 different commands during the session
- Log every command you run — the sequence itself is UX data.
  Patterns like "search → sql → expand" reveal missing output fields.
  Repeated patterns reveal automation opportunities.

## Evaluation

After exploration, read `references/evaluation.md` and produce the
full friction report with 5-dimension ratings.

## Persist Results

After producing the evaluation, save it to a permanent record.
This enables cross-session comparison and score trend tracking.

1. **Generate a slug** from the topic: lowercase, hyphens, max 40 chars.
   Example: "cytidine deaminase discovery" → `cytidine-deaminase`

1. **Write the session file** to `docs/dogfood/YYYY-MM-DD-{slug}.md`
   using the template in `references/evaluation.md` (see "Persistence
   Format" section). Include YAML frontmatter with scores and metadata.

1. **Update the index** at `docs/dogfood/README.md`: add a row to the
   session table with date, persona, topic, total score, and link.

1. **Cross-reference**: if the session led to design decisions (AD-\*),
   add the `design_refs` field in frontmatter. AD docs should link
   back using: `See [dogfood session](../../docs/dogfood/{filename})`.

1. **Update scenario coverage** in `docs/dogfood/scenarios.md`:
   change the matching scenario's status to `done` with session ID.

The session file must be **self-contained** — readable as a standalone
document without needing the conversation that produced it. Include
the full friction log, command sequence, and research findings.
