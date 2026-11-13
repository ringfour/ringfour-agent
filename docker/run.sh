#!/usr/bin/env bash
# Source ROS and the agent overlay, then run whatever was asked for.
#
#   docker run ... ringfour-agent                     # the agent, default profile
#   docker run ... ringfour-agent agent --profile X   # the agent, explicit
#   docker run ... ringfour-agent onboard --profile X # the onboarding tool
#   docker run ... ringfour-agent bash                # a shell, ROS sourced
#
# `agent` and `onboard` are the package's two console scripts; anything else
# is run verbatim, which is what makes `bash` and `ros2 topic list` work for
# debugging without a second image.
set -eo pipefail

# NOT in a subshell: `( source ... )` sets the variables in a child that
# exits immediately, so the exec below would run with no ROS on its
# PYTHONPATH. And not under `set -u` either — ROS's own setup scripts read
# unset variables (AMENT_TRACE_SETUP_FILES, COLCON_TRACE, ...) and would abort.
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source /opt/ringfour/setup.bash

case "${1:-}" in
    agent|onboard)
        command="$1"
        shift
        # Exec the real process, not ros2 run's child. Docker SIGTERM must
        # reach the agent so it can drain work and finish its boot marker.
        exec "/opt/ringfour/lib/ringfour_agent/${command}" "$@"
        ;;
    *)
        exec "$@"
        ;;
esac
