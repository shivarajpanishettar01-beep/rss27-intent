# Prompt 99 — audit before the paper uses any number

Audit the repo against docs/SPEC.md and CLAUDE.md:
1. For every number in reports/ and in outputs of scripts/make_*.py, trace it to the script, config hash and git hash that produced it.
2. Re-run every make_* script from a clean checkout and confirm identical outputs.
3. Confirm the passthrough check ran before every policy evaluation.
4. Confirm no session appears in two splits.
5. List every config value still null, and every place code would use a rig fact not marked known in docs/RIG_FACTS.md.
Write reports/AUDIT.md with findings ranked by severity. Fix nothing silently; propose fixes.
