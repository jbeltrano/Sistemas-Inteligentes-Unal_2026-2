"""
Punto de entrada del agente de Diamond Rush.

Flujo: pantalla -> (un screenshot) -> estado -> plan -> teclado.

Uso:
    python main.py                       # plan sobre captura.png
    python main.py --imagen levels/level1.png
    python main.py --capturar            # 1 screenshot de pantalla (mss)
    python main.py --enviar              # ademas suelta el plan por teclado
    python main.py --auto                # = --capturar --enviar

El plan se calcula con UNA captura (no se vuelve a mirar la pantalla
para decidir). --enviar solo simula el teclado; la verificacion de
progreso usa la misma imagen o, si se le pasa --capturar, recaptura
al final como control.
"""
import argparse

import config
from agente import Agent
from control import Teclado
from percepcion.captura import capturar_pantalla, leer_imagen, guardar_debug


def _parsear_region(texto: str):
    """'x,y,ancho,alto' -> tupla de enteros."""
    x, y, ancho, alto = (int(t) for t in texto.split(","))
    return (x, y, ancho, alto)


def _imprimir_plan(plan, estado) -> None:
    print(f"\n{'=' * 62}")
    print("PLAN RESUELTO")
    print(f"{'=' * 62}")
    print(f"personaje inicial: {estado.player_cell}")
    print(f"acciones: {plan.longitud} pasos  |  completo: {plan.completo}")
    if plan.pendientes:
        print(f"metas pendientes (inalcanzables con el modelo): {plan.pendientes}")
    if plan.mundo_final is not None:
        print(f"personaje final (predicho): {plan.mundo_final.personaje}")
        print(f"diamantes restantes (predichos): {len(plan.mundo_final.diamantes)}")

    if plan.acciones:
        print("\npasos:")
        for i, accion in enumerate(plan.acciones, 1):
            print(f"  {i:>4}. {accion.texto()}")
    print("=" * 62)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagen", metavar="RUTA", help="screenshot estatico (PNG/JPG)")
    parser.add_argument("--capturar", action="store_true", help="tomar 1 screenshot de la pantalla (mss)")
    parser.add_argument("--region", metavar="X,Y,ANCHO,ALTO", help="region de pantalla para --capturar")
    parser.add_argument("--enviar", action="store_true", help="simular el teclado (ejecutar el plan)")
    parser.add_argument("--auto", action="store_true", help="igual a --capturar --enviar")
    parser.add_argument("--paso", type=float, help="pausa entre teclas en segundos")
    parser.add_argument("--debug", action="store_true", help="imprimir el EstadoJuego completo")
    parser.add_argument("--no-verificar", action="store_true", help="no re-percibir al final")
    args = parser.parse_args()

    if args.auto:
        args.capturar = True
        args.enviar = True

    region = _parsear_region(args.region) if args.region else None

    # Percepcion: UN screenshot (archivo o pantalla).
    if args.imagen:
        entrada = leer_imagen(args.imagen)
    elif args.capturar:
        entrada = capturar_pantalla(region)
        guardar_debug(entrada, "captura_inicial.png")
    else:
        entrada = leer_imagen(config.IMAGEN_DEFECTO)

    # Capturador opcional SOLO para verificacion de progreso en vivo.
    capturador = None
    if args.enviar and args.capturar and not args.no_verificar:
        capturador = lambda: capturar_pantalla(region)

    teclado = Teclado(pausa_paso=args.paso) if args.enviar else None
    agente = Agent(
        teclado=teclado,
        capturador=capturador,
        imprimir_estado=args.debug,
    )

    # Decision: el camino solucion desde ese unico screenshot.
    plan = agente.compute(entrada)
    _imprimir_plan(plan, agente.estado)

    if args.enviar:
        if not args.capturar and not args.imagen:
            pass  # ya se decidio; ejecutar sobre la misma captura inicial
        print(f"\nEjecutando {plan.longitud} pasos por teclado...")
        agente.ejecutar(plan, pausa_paso=args.paso, verificar=not args.no_verificar)

    print("\n[fin]")


if __name__ == "__main__":
    main()