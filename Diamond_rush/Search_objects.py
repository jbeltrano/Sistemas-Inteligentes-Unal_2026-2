import cv2
from pathlib import Path

# ==========================================
# CONFIGURACIÓN
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

imagen_path = BASE_DIR / "captura.png"
template_path = BASE_DIR / "templates" / "personaje.png"
template_path = BASE_DIR / "templates" / "diamond.png"
template_path = BASE_DIR / "templates" / "door_key.png"
template_path = BASE_DIR / "templates" / "hueco.png"

UMBRAL = 0.80


# ==========================================
# CARGAR IMÁGENES
# ==========================================

imagen = cv2.imread(str(imagen_path))
template = cv2.imread(str(template_path))

if imagen is None:
    raise FileNotFoundError(
        f"No se pudo cargar la imagen: {imagen_path}"
    )

if template is None:
    raise FileNotFoundError(
        f"No se pudo cargar el template: {template_path}"
    )


# ==========================================
# TAMAÑO DEL TEMPLATE
# ==========================================

h, w = template.shape[:2]

print(f"Template: {w}x{h}")


# ==========================================
# TEMPLATE MATCHING
# ==========================================

resultado = cv2.matchTemplate(
    imagen,
    template,
    cv2.TM_CCOEFF_NORMED
)


# ==========================================
# OBTENER COORDENADAS
# ==========================================

ys, xs = (resultado >= UMBRAL).nonzero()

print(f"Coincidencias encontradas: {len(xs)}")


# ==========================================
# DIBUJAR RESULTADOS
# ==========================================

for x, y in zip(xs, ys):

    x = int(x)
    y = int(y)

    cv2.rectangle(
        imagen,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    cv2.putText(
        imagen,
        f"({x}, {y})",
        (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )


# ==========================================
# GUARDAR RESULTADO
# ==========================================

resultado_path = BASE_DIR / "resultado.png"

cv2.imwrite(str(resultado_path), imagen)

print(f"Resultado guardado en:")
print(resultado_path)
