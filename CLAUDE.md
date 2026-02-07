# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IsaacLabController is a Python package that enables real-time web-based control of IsaacLab physics simulations. It provides a FastAPI HTTP/WebSocket server that communicates with the simulation through an adapter pattern, allowing camera control, dynamic object management, material editing, and live video streaming from a web browser.

Documentation and comments are in Korean.

## Build & Install

```bash
pip install -e .                    # Standard editable install
pip install -e ".[dev]"             # With pytest and httpx
pip install -e ".[standalone]"      # With numpy/opencv (not needed inside IsaacLab env)
```

Requires Python >= 3.10. Uses `setuptools` with `src/` layout.

## Testing

```bash
pytest                              # Run all tests
pytest -v path/to/test_file.py      # Run a single test file
pytest -v path/to/test_file.py::test_name  # Run a single test
```

Dev dependencies: `pytest>=7.0.0`, `httpx>=0.24.0` (install via `.[dev]`).

## Architecture

### Adapter Pattern (core abstraction)

All simulation interaction goes through abstract base classes in `src/isaac_lab_controller/adapters/`:

- **SceneAdapter** (`base.py`) — Root interface. Returns sub-adapters, manages simulation lifecycle (`step()`, `reset()`, `is_running()`), and handles multi-camera selection.
- **CameraAdapter** (`camera_adapter.py`) — Camera pose control (`set_pose`, `set_lookat`, `orbit`), frame capture (`get_frame`), intrinsics.
- **ObjectAdapter** (`object_adapter.py`) — Dynamic object CRUD (`spawn`, `delete`, `set_pose`). Uses `ObjectInfo` dataclass.
- **MaterialAdapter** (`material_adapter.py`) — Material CRUD and binding to objects (`create`, `bind`, `unbind`, `update`). Uses `MaterialInfo` dataclass. Includes 7 preset materials.

Concrete implementations live in `src/isaac_lab_controller/examples/digital_twin/adapters.py` (wraps IsaacLab Camera, USD prims, UsdShade).

### Thread Safety: Command Queue Pattern

The server runs in a background thread while IsaacLab runs on the main thread (required for PyTorch/CUDA). All mutations use a **command queue**:

1. FastAPI route handlers validate requests and enqueue commands (no direct simulation calls).
2. The main simulation loop calls `process_commands()` on each adapter every frame to execute queued operations.
3. Frame capture (`update_frame()`) runs on the main thread and writes to a lock-protected buffer.

Key simulation loop pattern:
```python
while simulation_app.is_running():
    sim.step()
    scene.update(sim.get_physics_dt())
    adapter.get_camera_adapter().process_commands()
    adapter.get_object_adapter().process_commands()
    adapter.get_material_adapter().process_commands()
    adapter.get_camera_adapter().update_frame()
```

### Server Layer

`src/isaac_lab_controller/server/`:

- **ControlServer** (`app.py`) — Main entry point. Two modes: `run()` (blocking) or `start_background()` (non-blocking, for integration with simulation loops). Also exposes `create_app()` for standalone FastAPI usage.
- **WebSocketHandler** (`websocket_handler.py`) — `/ws/stream` for binary frame streaming, `/ws/control` for JSON control commands. Uses a frame queue (max 2) with overflow drop.
- **Routes** (`routes/`) — REST endpoints organized by domain: `camera.py`, `objects.py`, `materials.py`. All mounted under `/api/{domain}/`.

### IPC Bridge

`src/isaac_lab_controller/ipc/zmq_bridge.py` — ZMQ-based RPC bridge (`SimulationBridge` REP socket + `ServerBridge` REQ socket) enabling the web server and simulator to run in separate processes.

### Configuration

`src/isaac_lab_controller/utils/config.py` — `ControllerConfig` and `StreamingConfig` dataclasses, loadable from YAML via `load_config()`.

## API Endpoints

| Domain | Prefix | Key endpoints |
|---|---|---|
| Camera | `/api/camera` | `GET /list`, `POST /select`, `GET/POST /pose`, `POST /lookat`, `POST /orbit`, `GET /frame` |
| Objects | `/api/objects` | `GET /`, `GET /types`, `POST /spawn`, `DELETE /{id}`, `PATCH /{id}/pose` |
| Materials | `/api/materials` | `GET /`, `GET /presets`, `POST /`, `PATCH /{id}`, `POST /bind`, `POST /unbind/{id}` |
| WebSocket | `/ws` | `/ws/stream` (binary frames), `/ws/control` (JSON commands) |
