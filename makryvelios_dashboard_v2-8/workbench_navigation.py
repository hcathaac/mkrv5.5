"""Pure module history for Streamlit; never edits research data or approvals."""
from __future__ import annotations

HOME = '1. Executive overview'
PIPELINE = '25. Guided research pipeline'
HISTORY_KEY = '_v6_navigation_history'
LIMIT = 100


def observe(state, page, valid_pages):
    """Record actual navigation, not reruns; truncate the forward branch on a new visit."""
    if page not in valid_pages:
        raise ValueError('Unknown module')
    history, cursor = state.get(HISTORY_KEY, ([], -1))
    # Reset a stale history if modules were removed or the cursor is malformed.
    if not (-1 <= cursor < len(history)) or any(p not in valid_pages for p in history):
        history, cursor = [], -1
    if cursor >= 0 and history[cursor] == page:
        return history, cursor
    history = (history[:cursor+1] + [page])[-LIMIT:]
    cursor = len(history)-1
    state[HISTORY_KEY] = (history, cursor)
    return history, cursor


def navigate(state, action, valid_pages):
    """Called before widgets are constructed on the next rerun (on_click)."""
    history, cursor = state.get(HISTORY_KEY, ([], -1))
    if action in ('back', 'forward'):
        step = -1 if action == 'back' else 1
        index = cursor + step
        if not (0 <= index < len(history)) or history[index] not in valid_pages:
            return False
        target = history[index]
        state[HISTORY_KEY] = (list(history), index)
    elif action in ('home', 'pipeline'):
        target = HOME if action == 'home' else PIPELINE
        if target not in valid_pages:
            return False
        # observe() records these new destinations after the selection changes.
    else:
        raise ValueError('Unknown navigation action')
    state['_v6_family'] = 'All modules'
    state['_v6_module'] = target
    return True
