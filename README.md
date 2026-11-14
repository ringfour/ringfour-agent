# Ringfour agent

The ROS 2 bridge for the [Ringfour SDK](../ringfour-sdk). It preserves the A–C lifecycle, lease, reconciliation and recovery contracts. Robot specifics come from a reviewed profile.

Build from this repository with the SDK as an explicit named build context:

```sh
docker build --build-context sdk=../ringfour-sdk -f docker/Dockerfile -t ringfour-agent:c-preview .
docker run --rm ringfour-agent:c-preview agent --help
```

For Python packaging: `uv build`. `ringfour-agent` and `ringfour-onboard` are the installed commands; running them requires the ROS 2 environment. The nested `src/ringfour_agent` package remains colcon-installable; the image installs its ROS console scripts `agent` and `onboard`.

Mount a reviewed profile at `/etc/ringfour/profile.yaml` and durable state at `/var/lib/ringfour`. Configure ROS discovery for your deployment explicitly. Never discard an old agent's journal/quarantine state during migration. See [migration notes](../ringfour-testing/MIGRATION.md).

If a legacy checkpoint recorded a missing startup description as a model
change, an authenticated local operator may request `safety.recover` for
`restart-model`. This new recovery case succeeds only when the old checkpoint
had a known model hash, the startup observation was missing (not different),
and the currently observed model exactly matches that hash. The Agent must be
idle and inhibited. It removes only that model quarantine; the inhibit and any
unclean-restart quarantine remain, and no physical stop is asserted. A genuine
model change or unavailable model still requires investigation.

Tests and the dashboard live in [ringfour-testing](../ringfour-testing). This build does not launch or modify a robot or the SaaS stack.
