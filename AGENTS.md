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
Cuatro solvers Python de la misma tarea; la versión relevante para la entrega es `picas_fijas4.py`.
- `picas_fijas.py` — minimax+entropía vectorizado con numpy. **Interactivo**: juega contra un humano
  (`input()`), así que NO ejecutarlo en modo no interactivo. Precalcula una tabla numpy 5040×5040 al arrancar
  (lento, imprime progreso). Codifica el resultado como `picas*5+fijas`. Requiere `numpy` (única dependencia del repo).
- `picas_fijas2.py` — solo stdlib. Autocompetición con dos estrategias:
  primer candidato válido y minimax de Knuth.
- `picas_fijas3.py` — solo stdlib. Construye un árbol de decisión óptimo y lo cachea en
  `arbol_picas_fijas.json` (se genera en la primera ejecución; la construcción tarda ~2 min). El secreto
  de prueba está hardcodeado (actualmente `"1234"`).
- `picas_fijas4.py` — clase `PicasFijasAgent` con API tipo entorno-curso:
  `start()`, `try_attempt()` (Acción), `discover()` (responder a un intento del oponente),
  `feedBack()` (Percepción). Usa tabla numpy `uint8` (5040×5040) compartida entre instancias.
  **Benchmark medido (5040 secretos): promedio 5.34, peor caso 7** (el mejor del repo).
  Ctor `solo_candidatos=True` reproduce la lógica antigua (umbral 30; promedio 5.36, peor 8),
  solo p. comparar en el benchmark. Primer intento fijo `(5,6,7,8)` (probar `0123` no mejora).
  Tiene driver `__main__` de autojuego (inicialización ~2 s; ~0.1 s por partida).
- `benchmark_solvers.py` — benchmark de todos los solvers contra los 5040 secretos (promedio,
  mediana, peor caso, histograma, tiempo). Las estratégias `Minimax Knuth` y `Minimax+entropía`
  se toman por **muestra** (25 y 250) porque recalculan minimax por secreto (puede tardar varios min);
  el resto evalua los 5040 completos. Los emojis/acentos en `print` de los scripts fallan en consolas
  Windows cp1252 (`UnicodeEncodeError`); usar `PYTHONIOENCODING=utf-8` si aparece.

## Desarrollo
- No hay sistema de build ni de tests; verificaciones ad-hoc con `python Picas_y_Fijas/<script>.py`.
- `numpy` es la única dependencia externa (usada por `picas_fijas.py`, `picas_fijas4.py` y `benchmark_solvers.py`).

## Convenciones
- Documentación y commits en **español** (único idioma usado en el repo hasta ahora).
- Trabajar sobre la rama `main` (única rama existente).
- No commitear salvo que se pida explícitamente.