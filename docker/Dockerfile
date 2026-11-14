# syntax=docker/dockerfile:1
#
# The ringfour agent as a RELEASE ARTIFACT: ROS 2, the agent package and the
# `ringfour` SDK it imports, with nothing bind-mounted. Pushed to a registry
# and pulled onto a robot that has never seen this repo; the only things
# supplied at runtime are the profile YAML, the ROS graph and a state dir.
#
# Build from ringfour-agent with the SDK as an explicit named source context:
#
#   docker build --build-context sdk=../ringfour-sdk -f docker/Dockerfile -t ringfour-agent:preview .
#
# WHY ros-core AND NOT ros-base: the agent needs exactly one thing out of the
# 560 MB ros-base adds, tf2_ros_py, which is 3 MB on its own.
#
# WHY TWO STAGES: cmake, colcon, a C++ toolchain and pip are needed to build
# the ROS overlay and to resolve the SDK's pip dependencies, and none of them
# are needed to run it. The runtime stage is ros-core plus ~110 MB of apt.

ARG ROS_DISTRO=jazzy


# --------------------------------------------------------------- builder --
FROM ros:${ROS_DISTRO}-ros-core AS builder
ARG ROS_DISTRO
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# rosidl's generators and ament_cmake already ship in ros-core; this is the
# toolchain around them. python3-numpy is a BUILD dep here: rosidl_generator_py
# compiles a C extension against its headers.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        python3-colcon-common-extensions \
        python3-numpy \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# slam_toolbox service definitions, taken from the official package and
# rebuilt without the node — see slam_toolbox_shim/package.xml for why this
# is not simply `apt install ros-jazzy-slam-toolbox` (answer: 1.4 GB).
COPY docker/slam_toolbox_shim ./src/slam_toolbox
RUN apt-get update \
    && apt-get download "ros-${ROS_DISTRO}-slam-toolbox" \
    && dpkg-deb -x ros-"${ROS_DISTRO}"-slam-toolbox_*.deb /tmp/slam_toolbox \
    && cp -a "/tmp/slam_toolbox/opt/ros/${ROS_DISTRO}/share/slam_toolbox/srv" \
             ./src/slam_toolbox/srv \
    && rm -rf /tmp/slam_toolbox ros-*-slam-toolbox_*.deb /var/lib/apt/lists/*

COPY src/ringfour_agent ./src/ringfour_agent

# One overlay at /opt/ringfour holding both packages. --merge-install so the
# result is a single prefix that COPYs across in one line.
RUN source "/opt/ros/${ROS_DISTRO}/setup.bash" \
    && colcon build \
        --merge-install --install-base /opt/ringfour \
        --cmake-args -DCMAKE_BUILD_TYPE=Release

# The `ringfour` SDK and the two dependencies of it that Ubuntu ships too old
# to use (python3-pydantic is 1.10, we need >=2.5; python3-websockets is 10.4,
# we need >=12). They go into the overlay's own site-packages, so the single
# `source /opt/ringfour/setup.bash` in the entrypoint puts them on PYTHONPATH
# and there is no second install prefix to remember.
#
# --no-deps ON THE SDK IS LOAD-BEARING: its dependency list includes numpy, and
# an unconstrained pip resolves that to numpy 2.x. Shadowing ros-core's
# numpy 1.26 breaks rclpy's C extensions, which were compiled against it.
COPY --from=sdk . ./ringfour-sdk
RUN SITE="/opt/ringfour/lib/python$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')/site-packages" \
    && pip install --no-cache-dir --break-system-packages --target "${SITE}" \
        'pydantic>=2.5' 'websockets>=12' \
    && pip install --no-cache-dir --break-system-packages --target "${SITE}" \
        --no-deps ./ringfour-sdk \
    && rm -rf "${SITE}/bin"


# --------------------------------------------------------------- runtime --
FROM ros:${ROS_DISTRO}-ros-core AS runtime
ARG ROS_DISTRO
ENV DEBIAN_FRONTEND=noninteractive

LABEL org.opencontainers.image.title="ringfour-agent" \
      org.opencontainers.image.description="Robot-agnostic JSON-RPC bridge for ROS 2" \
      org.opencontainers.image.licenses="MIT"

# Message packages the agent imports but ros-core does not carry, plus the
# zenoh RMW: base101's Axon firmware is a zenoh-pico node, so a robot on
# rmw_zenoh needs the agent to be able to speak it (RMW_IMPLEMENTATION at
# runtime; FastDDS stays the default). ros-core already provides rclpy,
# common_interfaces, numpy and yaml.
#
# NOT HERE, DELIBERATELY:
#   nav2_map_server (+274 MB) — slam.py used to shell out to its
#     map_saver_cli for the grid export. It now renders grid.png + grid.yaml
#     itself, from the OccupancyGrid it already holds, pixel-for-pixel the
#     same as the CLI. Nothing left to install.
#   slam_toolbox (+1.4 GB) — the node runs in the robot's SLAM container. Only
#     its service definitions are here, from the builder stage.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        "ros-${ROS_DISTRO}-tf2-ros-py" \
        "ros-${ROS_DISTRO}-nav2-msgs" \
        "ros-${ROS_DISTRO}-control-msgs" \
        "ros-${ROS_DISTRO}-controller-manager-msgs" \
        "ros-${ROS_DISTRO}-rmw-zenoh-cpp" \
        "ros-${ROS_DISTRO}-rmw-cyclonedds-cpp" \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/ringfour /opt/ringfour

# Where the agent keeps what outlives a run: the audit log, saved maps,
# places. A volume mount point, so a container restart is not amnesia.
ENV RINGFOUR_STATE=/var/lib/ringfour
RUN mkdir -p "${RINGFOUR_STATE}" /etc/ringfour \
    && mkdir -p /run/ringfour-motion \
    && chown ubuntu:ubuntu "${RINGFOUR_STATE}" /run/ringfour-motion \
    && chmod 700 /run/ringfour-motion
VOLUME ["/var/lib/ringfour"]

COPY --chmod=0755 docker/run.sh /entrypoint.sh

# Non-root by default. `ubuntu` (uid 1000) already exists in the ROS image.
# This works across the /dev/shm payload handoff because the agent writes its
# segments 0644 — world-readable on purpose, so a client running as a
# different user on the host can mmap them (see ringfour_agent/shm.py).
USER ubuntu

EXPOSE 10101
ENTRYPOINT ["/entrypoint.sh"]
CMD ["agent", "--profile", "/etc/ringfour/profile.yaml"]
