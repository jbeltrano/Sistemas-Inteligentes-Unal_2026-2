"""
Memoria de botones y rejas del nivel.

Problema que resuelve: la percepcion es "sin memoria" (un frame -> un
GameState) y, justo cuando importa, pierde informacion:

  - un boton CUBIERTO (personaje o roca encima) deja de detectarse como
    `button`; si es el PRIMER frame del nivel (p. ej. el tester dejo al
    jugador sobre el boton antes de correr el agente), ni siquiera hay
    un frame anterior donde se viera: se reconstruye a partir de la
    reja abierta que su boton deja (ver _infer_pressed);
  - una reja ABIERTA se detecta como `open_gate`, y world_from_state la
    trata como piso (rol "floor"), asi que se olvida que existia y que
    se vuelve a cerrar al soltar su boton.

Al re-planificar desde un frame asi, el planificador recibia un mundo sin
ese boton y con esa reja convertida en piso permanente: planeaba cruzar
una reja que en el juego real se cierra. Esta clase guarda lo ESTATICO
(donde hay botones y rejas) desde el primer frame, que es el unico en que
suele estar todo descubierto, y lo reinyecta en cada World.

Ademas guarda lo APRENDIDO en ejecucion: que reja abrio realmente cada
boton (observacion al pisarlo) y si el fallo del juego (colarse por la
reja contigua al boton) funciono o no.

Este modulo importa SOLO de perception.state y button_gate (igual que
compute.py): nunca de cv2 ni de recognition.py.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple

import button_gate
import config
from perception.state import GameState

Cell = Tuple[int, int]


def _adjacent(a: Cell, b: Cell) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


@dataclass
class GateKnowledge:
    level: Optional[int] = None

    # --- estatico (se siembra con el primer frame) -------------------
    buttons: Set[Cell] = field(default_factory=set)
    gates: Set[Cell] = field(default_factory=set)  # abiertas o cerradas

    # --- aprendido en ejecucion --------------------------------------
    learned: Dict[Cell, Cell] = field(default_factory=dict)   # boton -> reja observada
    confirmed: Set[Cell] = field(default_factory=set)         # botones cuyo emparejamiento ya se verifico
    glitch_failed: Set[Cell] = field(default_factory=set)     # botones cuyo fallo NO funciono
    glitch_confirmed: Set[Cell] = field(default_factory=set)  # botones cuyo fallo SI funciono

    _seeded: bool = False

    # ---------------------------------------------------------------
    # Memoria estatica
    # ---------------------------------------------------------------
    def remember(self, state: GameState) -> None:
        """
        Siembra botones y rejas con el PRIMER frame del nivel y no vuelve a
        tocarlos. Es a proposito: si se acumularan en cada frame, un falso
        positivo del template matching quedaria como boton fantasma para
        siempre. Los botones y rejas no se mueven, asi que un solo frame
        sin nada encima basta.

        El hueco que llena: un boton CUBIERTO (por el personaje o por una
        roca, p. ej. si el tester dejo al jugador sobre el antes de correr
        el agente) no se detecta, y sin el el planificador trataba su
        reja como cerrada. Como una reja que se ve ABIERTA solo puede
        estarlo porque su boton este cubierto, _infer_pressed reconstruye
        ese par a partir de la reja.
        """
        if self._seeded:
            return
        self._seeded = True
        for obj in state.objects:
            cell = state.cell_of(obj.center)
            if obj.name == "button":
                self.buttons.add(cell)
            elif obj.name in ("closed_gate", "open_gate"):
                self.gates.add(cell)
        self._infer_pressed(state)

    def _infer_pressed(self, state: GameState) -> None:
        """
        Reconstruye el boton cubierto de una reja que se ve ABIERTA.

        La reja temporal solo esta abierta mientras su boton esta cubierto
        (el personaje o una roca encima); una reja abierta en el frame
        IMPLICA que su boton lo esta. Pero un boton cubierto no se
        detecta, asi que ese par se perdia y el modelo trataba la reja
        como cerrada (plan incompleto).

        Candidato a boton de la reja abierta G:
          1. El declarado en button_gate.py para este nivel cuya reja sea
             G (el confiable con varios pares); se exige que ese boton
             este realmente cubierto.
          2. Sin declaracion (o sin entrada para G): solo si es
             inequivoco -- UNA reja abierta y UNA celda cubierta. Con mas
             ambiguedad no se infiere nada (el lazo cerrado corrige).
        Y en ambos casos, consistencia: el candidato debe estar cubierto
        y no puede ser un boton VISIBLE (un boton descubierto mantendria
        su reja cerrada, seria inconsistente).

        Solo siembra `buttons`; NO toca `learned`, porque esto es una
        conjetura a partir del estado inicial, no evidencia observada al
        pisar un boton (meterlo en `learned` desplazaria un mapeo
        declarado correcto en _button_gate_map).
        """
        abiertas = {
            state.cell_of(o.center) for o in state.objects_named("open_gate")
        }
        if not abiertas:
            return

        cubiertas = {state.player_cell} | {
            state.cell_of(o.center) for o in state.objects_named("rock")
        }
        visibles = {
            state.cell_of(o.center) for o in state.objects_named("button")
        }
        declarado = (
            button_gate.BUTTON_GATE.get(self.level, {})
            if self.level is not None
            else {}
        )

        for gate in sorted(abiertas):
            candidato = next(
                (b for b, g in declarado.items() if g == gate), None
            )
            if candidato is None and len(abiertas) == 1 and len(cubiertas) == 1:
                # Sin declaracion, unica reja abierta y unica celda
                # cubierta: no hay otra explicacion posible.
                candidato = next(iter(cubiertas))
            if candidato in cubiertas and candidato not in visibles:
                self.buttons.add(candidato)

    # ---------------------------------------------------------------
    # Fallo del juego (colarse por la reja contigua al boton)
    # ---------------------------------------------------------------
    def active_glitch(self) -> Dict[Cell, Cell]:
        """
        Pares boton->reja en que se permite colarse. Solo los declarados a
        mano en button_gate.GATE_GLITCH para este nivel, contiguos, y que
        no hayan fallado ya en esta ejecucion.
        """
        if not config.USE_GLITCH or self.level is None:
            return {}
        declared = button_gate.GATE_GLITCH.get(self.level, {})
        return {
            b: g
            for b, g in declared.items()
            if _adjacent(b, g) and b not in self.glitch_failed
        }

    def disable_glitch(self, button: Cell) -> None:
        self.glitch_failed.add(button)

    # ---------------------------------------------------------------
    # Aprendizaje: que reja abrio un boton
    # ---------------------------------------------------------------
    @staticmethod
    def _cells(state: GameState, name: str) -> Set[Cell]:
        return {state.cell_of(o.center) for o in state.objects_named(name)}

    def observe_press(
        self, before: GameState, after: GameState, button: Cell
    ) -> Optional[Cell]:
        """
        Compara el frame de ANTES de pisar `button` con el de DESPUES y
        devuelve la reja que se abrio (y la guarda en `learned`). None si
        no hay evidencia clara (0 o varias rejas cambiaron): en ese caso no
        se aprende nada, para no reemplazar un mapeo bueno por un ruido.
        """
        player = after.player_cell
        opened = self._cells(after, "open_gate") - self._cells(before, "open_gate")
        if not opened:
            # Respaldo: la reja dejo de verse cerrada (la animacion puede no
            # haber alcanzado a mostrar el sprite abierto).
            opened = self._cells(before, "closed_gate") - self._cells(after, "closed_gate")
        opened.discard(player)
        if len(opened) != 1:
            return None
        gate = next(iter(opened))
        self.learned[button] = gate
        self.gates.add(gate)
        self.buttons.add(button)
        return gate


def glitch_candidates() -> Dict[int, Dict[Cell, Cell]]:
    """
    Ayuda de depuracion: pares boton->reja de BUTTON_GATE que son contiguos
    (donde el fallo del juego PODRIA aplicar). Es solo una pista para
    probar a mano; lo que de verdad se usa es button_gate.GATE_GLITCH.
    """
    found: Dict[int, Dict[Cell, Cell]] = {}
    for level, mapping in button_gate.BUTTON_GATE.items():
        pairs = {b: g for b, g in mapping.items() if _adjacent(b, g)}
        if pairs:
            found[level] = pairs
    return found
