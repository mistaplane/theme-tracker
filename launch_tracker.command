#!/bin/bash
# --- CONFIG ---
APP_DIR="/Users/mistaplane/Documents/scripts/theme_tracker"
PYTHON="/Library/Frameworks/Python.framework/Versions/3.13/bin/python3"
SCRIPT="$APP_DIR/tracker.py"
export THEMES_JSON="$APP_DIR/themes.json"
# -------------

cd "$APP_DIR"
"$PYTHON" "$SCRIPT"

