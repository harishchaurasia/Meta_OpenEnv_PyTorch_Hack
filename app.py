#!/usr/bin/env python3
"""Streamlit demo app for the Adaptive Navigation Environment."""

from __future__ import annotations

import streamlit as st

from adaptive_nav.env import ACTION_NAMES, AdaptiveNavEnv
from adaptive_nav.renderer import LEGEND, render_observation_rgb, render_rgb

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Adaptive Nav Env", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar -- configuration
# ---------------------------------------------------------------------------
st.sidebar.title("Configuration")
easy_mode = st.sidebar.checkbox("Easy map (7x7)", value=False)
grid_size = st.sidebar.slider(
    "Grid size", min_value=7, max_value=21, value=11, step=2,
    disabled=easy_mode,
)
view_radius = st.sidebar.slider("View radius", min_value=1, max_value=4, value=2)
max_energy = st.sidebar.slider("Max energy", min_value=20, max_value=200, value=80, step=10)
dynamic_changes = st.sidebar.checkbox("Dynamic changes", value=True)
dynamic_step = st.sidebar.slider("Dynamic step", min_value=5, max_value=60, value=25)
debug_mode = st.sidebar.checkbox("Debug mode (full grid + path hints)", value=False)
seed_input = st.sidebar.number_input("Seed (0 = random)", min_value=0, max_value=99999, value=0)
seed = seed_input if seed_input > 0 else None


def _make_env() -> AdaptiveNavEnv:
    env = AdaptiveNavEnv(
        grid_size=grid_size,
        view_radius=view_radius,
        max_energy=max_energy,
        dynamic_changes=dynamic_changes,
        dynamic_step=dynamic_step,
        seed=seed,
        debug=debug_mode,
        easy=easy_mode,
    )
    env.reset()
    return env


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "env" not in st.session_state:
    st.session_state.env = _make_env()
    st.session_state.history = []
    st.session_state.total_reward = 0.0
    st.session_state.last_info = {}

env: AdaptiveNavEnv = st.session_state.env

if st.sidebar.button("Reset Environment", use_container_width=True):
    st.session_state.env = _make_env()
    st.session_state.history = []
    st.session_state.total_reward = 0.0
    st.session_state.last_info = {}
    st.rerun()

# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------

def do_action(action: int) -> None:
    """Execute an action and store the result in session state."""
    e: AdaptiveNavEnv = st.session_state.env
    if e.is_done():
        return
    obs, reward, done, info = e.step(action)
    st.session_state.total_reward += reward
    st.session_state.last_info = info
    entry = f"Step {e.step_count}: {ACTION_NAMES[action]:>8s}  r={reward:+.1f}  {info.get('msg', '')}"
    if "dynamic_event" in info:
        entry += f"  [{info['dynamic_event']}]"
    st.session_state.history.append(entry)


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
st.title("Adaptive Navigation Environment")

col_grid, col_obs = st.columns([3, 2])

# --- Left column: full grid + action buttons ---
with col_grid:
    st.subheader("World")
    path_hints = env.get_path_hints() if debug_mode else None
    full_img = render_rgb(env.grid, env.agent_pos, cell_size=36, path_hints=path_hints)
    st.image(full_img, use_container_width=True)

    if env.is_done():
        reason = env.done_reason or "unknown"
        if env.won:
            st.success(f"You reached the goal!  (reason: {reason})")
        else:
            st.error(f"Episode over: {reason}")
    else:
        st.caption("Actions")
        btn_cols = st.columns(6)
        labels = ["Up", "Down", "Left", "Right", "Interact", "Wait"]
        for i, (col, label) in enumerate(zip(btn_cols, labels)):
            with col:
                if st.button(label, key=f"act_{i}", use_container_width=True):
                    do_action(i)
                    st.rerun()

# --- Right column: observation + mission + stats + log ---
with col_obs:
    # --- Mission panel ---
    st.subheader("Mission")
    mission = env.mission_status()
    m_cols = st.columns(3)
    for col, (label, done) in zip(m_cols, [
        ("Get Key", mission["get_key"]),
        ("Unlock Door", mission["unlock_door"]),
        ("Reach Goal", mission["reach_goal"]),
    ]):
        col.markdown(
            f"{'~~' if done else ''}**{'[x]' if done else '[ ]'} {label}**"
            f"{'~~' if done else ''}"
        )

    st.subheader("Agent Observation")
    obs = env.get_observation()
    obs_img = render_observation_rgb(obs["local_view"], cell_size=48)
    st.image(obs_img, use_container_width=True)

    st.subheader("Status")
    s1, s2, s3 = st.columns(3)
    s1.metric("Energy", f"{env.energy}/{env.max_energy}")
    s2.metric("Steps", env.step_count)
    s3.metric("Reward", f"{st.session_state.total_reward:+.1f}")

    s4, s5 = st.columns(2)
    s4.metric("Has Key", "Yes" if env.has_key else "No")
    s5.metric("Door", "Unlocked" if env.door_unlocked else "Locked")

    if debug_mode and "path_hints" in obs:
        st.subheader("Path Hints (debug)")
        for label, path in obs["path_hints"].items():
            st.caption(f"{label}: {len(path)} steps")

    st.subheader("Action Log")
    if st.session_state.history:
        log_text = "\n".join(reversed(st.session_state.history[-30:]))
        st.code(log_text, language=None)
    else:
        st.caption("No actions yet.")

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
with st.expander("Legend"):
    st.code(LEGEND, language=None)
