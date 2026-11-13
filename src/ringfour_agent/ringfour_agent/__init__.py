"""ringfour_agent: the ROS2 side of ringfour.

One generic node that loads a robot profile YAML and serves the ringfour
wire protocol (JSON-RPC 2.0 over WebSocket, unix socket + TCP).

Doctrine (spec section 24, discipline 10): this package knows no robot.
Everything robot-specific arrives via the profile. A robot-name
if-statement anywhere in here is a bug by definition.
"""
