#!/usr/bin/env bash
# Regenerate the GitHub Pages assets in docs/ from the live page in web/.
# Run after changing the paper or the stylesheet, then commit docs/.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

mkdir -p docs/img
cp web/style.css docs/paper.css
printf '/* Generated from web/style.css by tools/build_pages.sh — edit that file. */\n' |
  cat - docs/paper.css >docs/paper.css.tmp && mv docs/paper.css.tmp docs/paper.css

cp runs/main-hex/recon.png docs/img/recon.png

.venv/bin/python - <<'PY'
import pathlib
src = pathlib.Path("web/index.html").read_text()
start = src.index('<section class="paper" id="paper">')
end = src.index("</section>", start) + len("</section>")
paper = src[start:end].replace('src="/recon.png"', 'src="img/recon.png"')

template = pathlib.Path("tools/paper_template.html").read_text()
pathlib.Path("docs/paper.html").write_text(template.replace("{{PAPER}}", paper))
PY

echo "docs/paper.html and docs/paper.css rebuilt"
