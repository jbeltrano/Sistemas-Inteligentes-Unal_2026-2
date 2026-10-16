import sys
from pathlib import Path

# Mete la raíz de diamond_rush_v2 en sys.path para poder importar
# `config`, `perception...` y `computation...` corriendo pytest desde
# cualquier lado.
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
