import json
import os
import itertools
import random

def calcular_picas_y_fijas(intento, secreto):
    fijas = sum(1 for i in range(4) if intento[i] == secreto[i])
    picas = len(set(intento) & set(secreto)) - fijas
    return picas, fijas

class NodoArbol:
    def __init__(self, intento_optimo):
        self.intento = intento_optimo
        self.ramas = {}  # Diccionario {(picas, fijas): NodoArbol}

def construir_arbol_decision(posibilidades, universo):
    """Construye recursivamente el árbol de decisión óptimo."""
    if not posibilidades:
        return None
    if len(posibilidades) == 1:
        return NodoArbol(posibilidades[0])

    mejor_intento = None
    min_peor_caso = float('inf')
    candidatos = universo if len(posibilidades) > 15 else posibilidades

    for candidato in candidatos:
        grupos = {}
        for pos in posibilidades:
            res = calcular_picas_y_fijas(candidato, pos)
            grupos[res] = grupos.get(res, 0) + 1
        
        peor_caso = max(grupos.values())
        
        if peor_caso < min_peor_caso:
            min_peor_caso = peor_caso
            mejor_intento = candidato
        elif peor_caso == min_peor_caso and candidato in posibilidades:
            mejor_intento = candidato

    raiz = NodoArbol(mejor_intento)
    
    subgrupos = {}
    for pos in posibilidades:
        res = calcular_picas_y_fijas(mejor_intento, pos)
        subgrupos.setdefault(res, []).append(pos)
    
    for res, sub_posibilidades in subgrupos.items():
        if res == (0, 4):
            continue
        raiz.ramas[res] = construir_arbol_decision(sub_posibilidades, universo)
        
    return raiz

# --- FUNCIONES DE CONVERSIÓN A JSON (SERIALIZACIÓN) ---

def arbol_a_dict(nodo):
    """Convierte la estructura de objetos NodoArbol en un diccionario nativo de Python."""
    if nodo is None:
        return None
    
    dict_nodo = {
        "intento": nodo.intento,
        "ramas": {}
    }
    
    for (picas, fijas), sub_nodo in nodo.ramas.items():
        # JSON requiere que las llaves sean strings, guardamos como "picas,fijas" (ej: "1,2")
        llave_str = f"{picas},{fijas}"
        dict_nodo["ramas"][llave_str] = arbol_a_dict(sub_nodo)
        
    return dict_nodo

def dict_a_arbol(dict_nodo):
    """Reconstruye un objeto NodoArbol a partir de un diccionario cargado del JSON."""
    if dict_nodo is None:
        return None
    
    nodo = NodoArbol(dict_nodo["intento"])
    
    for llave_str, sub_dict in dict_nodo["ramas"].items():
        # Convertimos la llave "picas,fijas" de regreso a una tupla de enteros (picas, fijas)
        picas, fijas = map(int, llave_str.split(","))
        nodo.ramas[(picas, fijas)] = dict_a_arbol(sub_dict)
        
    return nodo

# --- FUNCIÓN DE JUEGO ---

def resolver_con_arbol(nodo_actual, secreto, intentos_realizados=1):
    """Navega por el árbol precomputado de manera instantánea."""
    intento = nodo_actual.intento
    picas, fijas = calcular_picas_y_fijas(intento, secreto)
    print(f"Intento {intentos_realizados}: {intento} -> Picas: {picas}, Fijas: {fijas}")
    
    if fijas == 4:
        print(f"🎉 ¡El árbol resolvió el juego en {intentos_realizados} intentos!\n")
        return intentos_realizados
    
    siguiente_nodo = nodo_actual.ramas.get((picas, fijas))
    if siguiente_nodo:
        return resolver_con_arbol(siguiente_nodo, secreto, intentos_realizados + 1)
    else:
        print("❌ Error: Respuesta inconsistente.")
        return intentos_realizados


def generar_primer_intento():
    lista = list()
    for i in range(0,4):
        num_generated = random.randint(0,9)
        while num_generated in lista:
            num_generated = random.randint(0,9)
        
        lista.append(num_generated)
    
    text = str()
    for i in lista:
        text += str(i)
        
    return text


# --- FLUJO PRINCIPAL ---

if __name__ == "__main__":
    ARCHIVO_JSON = "arbol_picas_fijas.json"
    arbol_perfecto = None

    # 1. Intentar cargar desde el archivo para ahorrar tiempo
    if os.path.exists(ARCHIVO_JSON):
        print("⚡ Cargando Árbol de Decisión desde el archivo JSON (Instantáneo)...")
        with open(ARCHIVO_JSON, "r", encoding="utf-8") as f:
            datos_json = json.load(f)
        arbol_perfecto = dict_a_arbol(datos_json)
        print("🚀 ¡Árbol cargado con éxito en 0 milisegundos!")
    else:
        # Si el archivo no existe, lo calcula por primera y única vez
        print("⏳ Archivo JSON no encontrado. Generando árbol por primera vez (tardará unos segundos)...")
        universo = ["".join(p) for p in itertools.permutations("0123456789", 4)]
        arbol_perfecto = construir_arbol_decision(universo, universo)
        
        # Guardar en el archivo JSON para el futuro
        print("💾 Guardando estructura óptima en 'arbol_picas_fijas.json'...")
        dict_para_json = arbol_a_dict(arbol_perfecto)
        with open(ARCHIVO_JSON, "w", encoding="utf-8") as f:
            json.dump(dict_para_json, f, indent=4)
        print("✅ Archivo guardado. Las próximas ejecuciones serán instantáneas.\n")

    # 2. Probar la velocidad del árbol resolviendo un número secreto
    numero_secreto = "5132"
    print(f"\n--- Iniciando Búsqueda en Árbol para: {numero_secreto} ---")
    resolver_con_arbol(arbol_perfecto, numero_secreto)
