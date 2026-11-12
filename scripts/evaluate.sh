#!/bin/bash
# Standard evaluation entry point: scores every method under outputs/ on the
# visible development scenes and writes scoring/scores.json (same content as
# the train slice). It reads only outputs/<method>/predictions.pt and
# evaluator-owned targets; nothing in method/ is imported.
#
# Prediction contract (outputs/<method>/predictions.pt, torch.save of a dict):
#   {"novel_view":         {scene_id: float [6, 3, 96, 64]  right halves of held-out captures},
#    "average_appearance": {scene_id: float [6, 3, 96, 128] render without per-capture appearance state}}
# for every scene in data/glav_syn/manifest.json, values in [0, 1].
set -e
cd /home/user
mkdir -p scoring
python -m eval.train.evaluate --outputs outputs --scores scoring/scores.json
