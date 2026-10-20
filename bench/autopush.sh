#!/bin/bash
# Every 10 minutes: if results changed, regenerate docs/samples.md and the Phase-1 table, commit and push. Stops after phase5_code done.
cd "$(dirname "$0")/.."; PY=.venv/bin/python; LOG=results/phase1_runs.log
while true; do
  if [ -n "$(git status --porcelain results docs 2>/dev/null)" ] || [ -n "$(git status --porcelain results)" ]; then
    $PY bench/make_samples.py > /dev/null 2>&1
    { sed -n '1,/^## Results/p' docs/phase1.md; echo; echo "Measured on GPU 5 (one A100 80GB PCIe), 256 generated tokens, 10 EN/JA prompts, greedy; mean over prompts. GSM8K-50 accuracy from results/gsm8k50_*.json. Updated automatically as runs complete."; echo; $PY bench/summarize.py; echo; echo "### GSM8K-50 accuracy (greedy)"; for f in results/gsm8k50_*.json; do [ -s "$f" ] && $PY -c "import json,sys; r=json.load(open('$f')); a=r.get('accuracy') or {}; s=r['summary']; print('* %s: %.0f%% (%d), %.1f tok/s effective' % ('$f'.split('/')[-1], 100*a.get('accuracy',0), a.get('n',0), s['tok_s_effective']['mean']))"; done; echo; echo "### HumanEval pass@1"; for f in results/humaneval_*.json; do [ -s "$f" ] && $PY -c "import json; r=json.load(open('$f')); print('* %s: %.1f%% (%d problems)' % ('$f'.split('/')[-1], 100*r['pass@1'], r['n']))"; done; } > docs/phase1.tmp && mv docs/phase1.tmp docs/phase1.md
    git add -A results docs > /dev/null 2>&1
    git -c user.name=shi3z -c user.email=shi3z@zelpm.com commit -q -m "results: automatic update $(date '+%Y-%m-%d %H:%M')" > /dev/null 2>&1 && git push -q origin main > /dev/null 2>&1 && echo "$(date +%T) pushed" >> results/autopush.log
  fi
  grep -q "phase5_code done" $LOG 2>/dev/null && break
  sleep 600
done
echo "$(date +%T) autopush finished" >> results/autopush.log
