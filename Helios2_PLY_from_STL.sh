#!/bin/bash

BLENDER_PATH="/Applications/Blender.app/Contents/MacOS/Blender"
PYTHON_SCRIPT="blend_utils_2.py"
STL_FOLDER="data/structured_dataset/STL"
OUT_DIR="test_sample/test_output"

"$BLENDER_PATH" --background --python "$PYTHON_SCRIPT" -- --stl_folder "$STL_FOLDER" --out_dir "$OUT_DIR"