# AGENTS.md

Repositorio del curso (Sistemas Inteligentes, UNAL 2026-2). Contiene proyectos independientes, una carpeta por proyecto. No hay tooling compartido: ni build/test, ni linter, ni framework de tests, ni CI, ni `.gitignore`, ni `requirements`.

## Proyectos

- `Picas_y_Fijas/` — Proyecto 1 (agente para juego mastermind-style). Efectivamente terminado: `picas_fijas.py` (solver tipo brute-force/información, único que usa numpy), `picas_fijas2.py` (minimax/Knuth, solo stdlib) y `picas_fijas3.py` (árbol de decisión precomputado, solo stdlib).
- `Diamond_rush/` — Proyecto 2 (activo). Meta (README.md): un agente que juegue los 20 niveles de "Diamond Rush" en https://www.minijuegos.com/juego/diamond-rush. Etapa actual: *solo detección de objetos y camino*; aún NO es agente jugador (sin captura de pantalla en vivo ni control de entrada).

## Comandos

Correr los scripts desde dentro de su propia carpeta:

```
# Picas_y_Fijas
python picas_fijas.py     # picas_fijas2.py / picas_fijas3.py equivalentes

# Diamond_rush (requiere opencv-python)
python Search_objects.py  # lee captura.png, dibuja coincidencias, escribe resultado.png y caminos.png
```

## Convenciones y gotchas (Diamond_rush)

- `Search_objects.py` usa `cv2.matchTemplate` (TM_CCOEFF_NORMED) sobre 14 tipos de objeto. El personaje usa `reflejar=True` (prueba también el template espejado horizontal). NMS greed y con `SOLAPE_MAX` (fracción del área del template) para una sola caja por objeto.
- El umbral está seteado dos veces en el bloque CONFIGURACIÓN (`UMBRAL = 0.80` seguido de `UMBRAL = 0.90`): manda el último. `SOLAPE_MAX = 0.4` es la fracción máxima del área del template compartida entre dos coincidencias antes de considerarlas la misma meseta.
- Además de objetos, detecta el suelo/camino por color HSV (`detectar_caminos`, escribe `caminos.png`) y expone `crear_grid()` que convierte la máscara a una grilla transitable (casilla = camino si >50% de píxeles >0). Los rangos HSV del suelo son explícitamente "un punto de partida" a calibrar viendo la máscara resultante.
- El input por defecto es `captura.png`. Para probar un nivel concreto hay capturas estáticas en `levels/level1..20.png`; se cambia descomentando la línea `imagen_path = BASE_DIR / "levels" / "level19.png"` en el bloque CONFIGURACIÓN.
- Los templates de `templates/` son screenshots recortados a mano de un nivel/frame específico: si el juego se ve distinto, dejan de matchear. No hay 1:1 entre los templates y el código: `totem.png` no se usa en la lista de objetos y la variable `space_enable_path` apunta a `scape_enable.png`.
- Regla anti-scraping: el agente solo lee píxeles del juego (imágenes/pantalla); nunca HTTP/DOM/JS del sitio.

## Artefactos y estado del repo

- `picas_fijas3.py` cachea su árbol a `arbol_picas_fijas.json` en el cwd (artefacto generado, no commitear). En Diamond_rush los outputs generados son `resultado.png` y `caminos.png` (commiteados hoy mismo; van y vienen con cada experimento), y `captura.png` es el sample de entrada.
- Dependencias actuales: `numpy` (solo `picas_fijas.py`) y `opencv-python` (Diamond_rush).
- Git: `main` está al día con `origin/main`; hay una rama `origin/nuevo_picas_fijas` (intento alterno de picas y fijas). El working tree suele tener cambios sin commitear en `Diamond_rush/` — revisa `git status` antes de commitear.
- Todo texto legible por humanos va en español: READMEs, comentarios, prints de consola y mensajes de commit.