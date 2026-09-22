# AGENTS.md

Course repo (Sistemas Inteligentes, UNAL 2026-2). It contains independent projects, one folder each. There is no shared build/test tooling, no linter, no test framework, no CI, and no `.gitignore`.

## Projects

- `Picas_y_Fijas/` — Project 1 (agent for mastermind-style game). Effectively done: a pure-numpy brute-force/information-theory solver (`picas_fijas.py`), a plain-Python minimax/Knuth solver (`picas_fijas2.py`), and a precomputed decision-tree solver (`picas_fijas3.py`).
- `Diamond_rush/` — Project 2 (active). Goal (README.md): an agent that plays the 20 levels of "Diamond Rush" at https://www.minijuegos.com/juego/diamond-rush. Current stage is *object detection only*: `Search_objects.py` runs `cv2.matchTemplate` (TM_CCOEFF_NORMED, threshold 0.80) to locate game objects in the static screenshot `captura.png` using PNG crop templates in `templates/`, keeps one box per object via greedy non-maximum suppression (`SOLAPE_MAX`, a fraction of the template area, in the CONFIGURACIÓN block), and writes `resultado.png`. It is NOT yet a player agent and does no screen capture / input control.

## Commands

Run scripts from inside their own folder:

```
# Picas_y_Fijas
python picas_fijas.py     # picas_fijas2.py / picas_fijas3.py equivalentes

# Diamond_rush (requiere opencv-python)
python Search_objects.py  # lee captura.png, dibuja coincidencias sobre los templates en templates/, escribe resultado.png
```

## Conventions and gotchas

- All human-facing text is in Spanish: READMEs, code comments, console `print()` output, and git commit messages. Match this in new code.
- Anti-scraping rule (Diamond_rush): the agent only reads pixels from the game screen/images; never HTTP/DOM/JS of the game site.
- `picas_fijas3.py` caches its computed decision tree to `arbol_picas_fijas.json` in the current working directory (generated artifact, don't commit). `Diamond_rush` templates are hand-cropped screenshots of a specific level/frame; `captura.png` and `resultado.png` are the current sample input/output and drift with each experiment (they are committed today).
- Git state: branch `main` has diverged from `origin/main` (1 local vs 3 remote commits). `Picas_y_Fijas/` and most of `Diamond_rush/` are tracked; `AGENTS.md` and `templates/no_picos.png` are untracked. Mind the diverged branch before pushing.
- Current deps: `numpy` (only `picas_fijas.py`; the other two solvers are stdlib-only) and `opencv-python` (Diamond_rush). No requirements files exist.