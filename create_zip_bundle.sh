#!/usr/bin/env bash
set -euo pipefail

timestamp="$(date +%Y%m%d_%H%M%S)"
target="/home/ycy/sdi/Skill_Evo/skill_bundle_${timestamp}.zip"

sources=(
  "home/ycy/sdi/skill_saved/embedding_cache"
  "home/ycy/sdi/data/DeepMath-103K.jsonl"
  "home/ycy/sdi/data/temp_data.jsonl"
  "home/ycy/sdi/data/temp_data.parquet"
  "home/ycy/sdi/data/greedy_data.jsonl"
  "home/ycy/sdi/data/greedy_data.parquet"
  "home/ycy/sdi/Skill_Evo/baselines/SkillRL/outputs"
)

cd /
zip -r "$target" "${sources[@]}"
printf '%s\n' "$target"
