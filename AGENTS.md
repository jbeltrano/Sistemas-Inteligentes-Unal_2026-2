# AGENTS.md

## Proyecto
Curso **Sistemas Inteligentes** (UNAL, 2026-2), Proyecto 1: agente inteligente que juega
**Picas y Fijas** en la menor cantidad de intentos. Todo el trabajo vive en `Picas_y_Fijas/`.

## Arquitectura requerida (ver `Picas_y_Fijas/README.md`)
- Agente **utilitario** (Utility-Based Agent), no basado en reglas.
- Flujo interno: `Percepción -> Computación -> Acción`.
- El medio ambiente entrega la información (percepción) que se usa para calcular.

## Reglas del juego (estándar colombiano)
- Número secreto de **4 dígitos sin repetir** generado por el medio ambiente.
- Cada intento devuelve: `fijas` = dígitos correctos en posición correcta;
  `picas` = dígitos correctos en posición incorrecta.
- Verificar el protocolo exacto del curso si difiere (p. ej. repetición de dígitos).

## Código (`Picas_y_Fijas/`)
- `picas_fijas4.py` — clase `PicasFijasAgentCJBVC` con API tipo entorno-curso:
  `start()`, `try_attempt()` (Acción), `discover()` (responder a un intento del oponente),
  `feedBack()` (Percepción). Usa tabla numpy `uint8` (5040×5040) compartida entre instancias.
  El criterio de computación es configurable con `estrategia=`:
  - `entropia` (por defecto): maximiza entropía, desempata por peor caso y ser candidato.
    **Benchmark completo (5040 secretos) — promedio 5.2470, peor caso 7**.
  - `esperado`: minimiza el tamaño esperado de candidatos restantes (suma de cuadrados).
    Promedio 5.2651, peor caso 7. Queda un poco por detrás de `entropia`.
  - `minimax_entropia`: reproduce la versión antigua (minimiza peor caso primero).
    Promedio 5.3373, peor caso 8.
  - `profundidad2`: lookahead de 2 niveles por costo esperado con beam + muestreo reducido
    (variante experimental; más lenta y sin ventaja medida frente a `entropia`).
  El peor caso 7 es el mínimo demostrado para este juego y el promedio óptimo teórico es
  ≈5.21 (Wikipedia: 26274/5040), así que queda poco margen. Caché por nodo (`CACHE`,
  clave = conjunto de candidatos + estrategia) reutiliza decisiones entre partidas y acelera
  los benchmarks (el completo con caché tarda ~11 s). Ctor `solo_candidatos=True` restringe
  la búsqueda a candidatos cuando quedan pocos (umbral 30; empeora el promedio, solo p.
  comparar). Primer intento fijo `(5,6,7,8)` (probar `0123` no mejora; por simetría de
  dígitos todos los primeros intentos equivalen). Tiene driver `__main__` de autojuego
  (inicialización ~2 s) que acepta la estrategia por argv.
- `benchmark.py` — benchmark del agente contra los 5040 secretos (promedio, mediana, peor caso,
  histograma, tiempo). Uso: `--muestra N` (muestra aleatoria con semilla fija 12345, para iterar
  rápido) o `--completo` (los 5040, ~11 s con caché). Opciones: `--estrategia`,
  `--solo-candidatos`, `--umbral`, `--sin-cache`. Los prints son ASCII
  (los acentos/emojis de scripts causan `UnicodeEncodeError` en consolas Windows cp1252;
  usar `PYTHONIOENCODING=utf-8` si aparece).
- `competencia.py` — torneo cara a cara `dinoAgent` vs `PicasFijasAgentCJBVC`: saca `--partidas N`
  (default 100, semilla fija 12345) y `--detalle` (imprime la traza intento→feedback de cada
  partida). Verifica por partida que cada agente cuente solo los intentos hasta que ÉL acierta.

## Desarrollo
- No hay sistema de build ni de tests; verificaciones ad-hoc:
  `python Picas_y_Fijas/benchmark.py --muestra 500`, `python Picas_y_Fijas/picas_fijas4.py`
  y `python Picas_y_Fijas/competencia.py`.
- `numpy` es la única dependencia externa (usada por `picas_fijas4.py` y `benchmark.py`).

## Convenciones
- Documentación y commits en **español** (único idioma usado en el repo hasta ahora).
- Trabajar sobre la rama `main` (única rama existente).
- No commitear salvo que se pida explícitamente.