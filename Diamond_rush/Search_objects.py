import cv2
from pathlib import Path
from random import random
import numpy as np

class MatchTemplate:
    
    def __init__(self, template_path, imagen, reflejar=False):
        self.imagen = imagen
        self.template_path = template_path
        self.reflejar = reflejar
        
        self.template_image = cv2.imread(str(self.template_path))
        self.check_images_exist(self.template_image, self.template_path)
        
        self.get_template_size()
        
        
    @staticmethod
    def check_images_exist(imagen, path):
        if imagen is None:
            raise FileNotFoundError(
                f"No se pudo cargar la imagen: {path}"
            )
    
    def get_template_size(self):
        h, w = self.template_image.shape[:2]
        print(f"Template: {w}x{h}")
        return h, w
    
    def match_template(self, threshold=0.80, solape_max=None):
        if solape_max is None:
            solape_max = SOLAPE_MAX
        resultado = cv2.matchTemplate(
            self.imagen,
            self.template_image,
            cv2.TM_CCOEFF_NORMED
        )
        
        # Si el objeto puede estar girado, probar también el template reflejado
        # y quedarse con el mejor score de ambas orientaciones
        if self.reflejar:
            resultado_flip = cv2.matchTemplate(
                self.imagen,
                cv2.flip(self.template_image, 1),
                cv2.TM_CCOEFF_NORMED
            )
            resultado = np.maximum(resultado, resultado_flip)
        
        ys, xs = (resultado >= threshold).nonzero()
        scores = resultado[ys, xs]
        h, w = self.get_template_size()
        
        # NMS: quedarse con una sola caja por objeto
        # (matchTemplate deja una "meseta" de píxeles sobre el umbral alrededor de cada coincidencia)
        candidatos = sorted(
            zip(xs, ys, scores),
            key=lambda p: p[2],
            reverse=True
        )
        
        elegidos = []
        for x, y, _ in candidatos:
            if all(
                self._sobrelape(x, y, ex, ey, w, h) <= solape_max
                for ex, ey in elegidos
            ):
                elegidos.append((x, y))
        
        xs, ys = zip(*elegidos) if elegidos else ([], [])
        print(f"Coincidencias encontradas: {len(elegidos)}")
        
        return list(xs), list(ys)
    
    def _sobrelape(self, x, y, dx, dy, w, h):
        ancho = max(0, min(x + w, dx + w) - max(x, dx))
        alto  = max(0, min(y + h, dy + h) - max(y, dy))
        return (ancho * alto) / (w * h)
    
# ==========================================
# CONFIGURACIÓN
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

imagen_path = BASE_DIR / "captura.png"
# imagen_path = BASE_DIR / "levels" / "level19.png"
template_path = BASE_DIR / "templates" / "personaje.png"
personaje_path = BASE_DIR / "templates" / "personaje.png"
diamond_path = BASE_DIR / "templates" / "diamond.png"
door_key_path = BASE_DIR / "templates" / "door_key.png"
hueco_path = BASE_DIR / "templates" / "hueco.png"
key_path = BASE_DIR / "templates" / "key.png"
pic_path = BASE_DIR / "templates" / "pic.png"
rock_in_path = BASE_DIR / "templates" / "rock_in.png"
rock_path = BASE_DIR / "templates" / "rock.png"
scape_block_path = BASE_DIR / "templates" / "scape_block.png"
space_enable_path = BASE_DIR / "templates" / "scape_enable.png"
no_picos_path = BASE_DIR / "templates" / "no_picos.png"
button_path = BASE_DIR / "templates" / "button.png"
lava_path = BASE_DIR / "templates" / "lava.png"
reja_path = BASE_DIR / "templates" / "reja.png"

UMBRAL = 0.80
UMBRAL = 0.90
# UMBRAL = 0.96
# Fracción máxima del área del template que dos coincidencias pueden compartir
# antes de considerarse la misma meseta (0 = nada, 1 = todo)
SOLAPE_MAX = 0.4

imagen = cv2.imread(str(imagen_path))
MatchTemplate.check_images_exist(imagen, imagen_path)

objects = [
    MatchTemplate(personaje_path, imagen, reflejar=True), 
    MatchTemplate(diamond_path, imagen), 
    MatchTemplate(door_key_path, imagen), 
    MatchTemplate(hueco_path, imagen), 
    MatchTemplate(key_path, imagen), 
    MatchTemplate(pic_path, imagen), 
    MatchTemplate(rock_in_path, imagen), 
    MatchTemplate(rock_path, imagen), 
    MatchTemplate(scape_block_path, imagen), 
    MatchTemplate(space_enable_path, imagen),
    MatchTemplate(no_picos_path, imagen),
    MatchTemplate(button_path, imagen),
    MatchTemplate(lava_path, imagen),
    MatchTemplate(reja_path, imagen)
    ]


# Obteiene los resultados 
data = []
for obj in objects:
    
    xs, ys = obj.match_template(threshold=UMBRAL)
    h, w = obj.get_template_size()
    
    r = int(random() * 255)
    g = int(random() * 255)
    b = int(random() * 255)
    data.append((xs,ys,h,w,r,g,b))
    

# Dibuja los resultados en la imagen
for xs, ys, h, w, r, g, b in data:
    for x, y in zip(xs, ys):

        x = int(x)
        y = int(y)

        cv2.rectangle(
            imagen,
            (x, y),
            (x + w, y + h),
            (r, g, b),
            2
        )

        cv2.putText(
            imagen,
            f"({x}, {y})",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (r, g, b),
            1
        )


resultado_path = BASE_DIR / "resultado.png"

cv2.imwrite(str(resultado_path), imagen)

print(f"Resultado guardado en:")
print(resultado_path)
