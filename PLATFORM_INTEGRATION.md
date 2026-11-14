# Ringfour Platform integration note

The sibling `../platform` project is developing Logbook (milestone 08). An
experimental `motion_journal.py` implements an independent, local SQLite
journal for the proposed selective motion records. An optional dedicated
local socket exposes journal reads and acknowledgements; it does not expose
robot control and is disabled by default. Selected task, teleop, measured-stop
and guard outcomes are recorded when enabled. No Platform service or
configuration is needed to run the Agent.

The Agent remains responsible for its existing ROS bridge and reviewed robot
profiles. Inspect and document the actual Agent protocol and version in Platform
before connecting Agent-originated information; do not infer an interface from
the Platform event schema. Preserve
persistent Agent journal and quarantine state during any deployment migration.

See `../platform/docs/logbook-contract.md` and
`../platform/docs/evidence/08.md` for the current proposal and test status.

The desired Platform Logbook selectively retains motion lifecycle and outcomes:
accepted/refused, completed/cancelled, verified stops and guard trips. It must
summarize a continuous drive session rather than replicate each velocity update.
The current SDK audit query reads an in-memory ring and audit tail is live-only;
the Agent also writes rotating local JSONL. `AuditEvent.client` identifies an
Agent connection, not inherently a Platform execution or human. A durable
handoff exists, but coverage of all motion paths and trustworthy Platform
owner/execution correlation remain incomplete. Do not interpret missing
records as proof of a verified stop.

The local journal's synchronous writes are not safe on a physical stop path.
The producer enqueues in memory and writes on a separate worker. Overflow and
unclean boots are reported as `motion.history_incomplete` when storage recovers;
terminal-event capacity is not reserved. The daemon can replicate the stream to
Platform history. Offline tests use synthetic stop evidence, not a real robot.

The Agent and SDK must remain independently usable in the open-source stack
without any Platform service or configuration. A future Logbook handoff must be
optional and must never make motion or stop behavior depend on the Platform.
