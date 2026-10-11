"""
Contrato del test suite: asegura que Diamond_rush quede en sys.path
sin importar desde donde se invoque pytest.
"""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))