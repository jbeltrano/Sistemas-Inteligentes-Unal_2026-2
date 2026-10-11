"""
Tests de integracion del agente: perception + decision sobre la
captura real del repo (requiere OpenCV, numpy y los templates).
"""
import config
from agente import Agent


def test_compute_sobre_captura_real():
    """Agent.compute() resuelve un plan desde el screenshot estatico."""
    agente = Agent()  # sin teclado: solo se usa al ejecutar
    plan = agente.compute(config.IMAGEN_DEFECTO)

    assert agente.estado is not None
    assert agente.mundo is not None
    assert plan.mundo_inicial is agente.mundo
    assert isinstance(plan.acciones, list)
    assert plan.completo or plan.pendientes  # o resuelto, o reporta pendientes
    # En la captura del repo hay diamantes que recolectar: hay plan.
    assert plan.longitud > 0


def test_compute_reporta_pendientes_sin_romper():
    """Una meta inalcanzable no tira el agente: queda listada."""
    agente = Agent()
    plan = agente.compute(config.IMAGEN_DEFECTO)

    if plan.pendientes:
        assert not plan.completo
    else:
        # Si no hay pendientes, el plan debe ser completo.
        assert plan.completo