"""Models the AGENT owns.

The profile schema lives here, not in the SDK, because the SDK never reads
a profile: verified 2026-08-26, no module under `ringfour/ringfour/`
imports `RobotProfile`, `GroupConfig`, `ToolConfig` or any other name from
it. Only the agent parses YAML.

The distinction that matters is between two things that both say
"profile":

* `ringfour.models.protocol.ProfileInfo` — the WIRE type. Name, model,
  version, capabilities; the handful of fields a client is told at
  handshake. Stays in the SDK, because clients read it.
* `ringfour_agent.models.profile.RobotProfile` — the FILE schema. 564
  lines of vocabulary for describing a robot. A mission author never
  writes one; a robot integrator does. Shipping it in the client library
  meant every mission author downloaded the grammar for configuring
  machines they do not configure.
"""

from .profile import *          # noqa: F401,F403
from . import profile           # noqa: F401
