import itertools
import random

def calcular_picas_y_fijas(intento, secreto):
    """Calcula el número de picas y fijas entre dos códigos de 4 dígitos."""
    fijas = sum(1 for i in range(4) if intento[i] == secreto[i])
    # Las picas son los dígitos en común menos las fijas
    picas = len(set(intento) & set(secreto)) - fijas
    return picas, fijas

def estrategia_simple(secreto, primer_intento="1234"):
    """Resuelve el juego usando la heurística del primer candidato válido."""
    # Generar todas las 5,040 combinaciones de 4 dígitos únicos
    posibilidades = ["".join(p) for p in itertools.permutations("0123456789", 4)]
    
    intento = primer_intento
    intentos_realizados = 0
    
    print(f"--- Iniciando Estrategia Simple para el secreto: {secreto} ---")
    
    while True:
        intentos_realizados += 1
        picas, fijas = calcular_picas_y_fijas(intento, secreto)
        print(f"Intento {intentos_realizados}: {intento} -> Picas: {picas}, Fijas: {fijas}")
        
        if fijas == 4:
            print(f"🎉 ¡Ganó en {intentos_realizados} intentos!\n")
            return intentos_realizados
            
        # Filtrar (podar) el espacio de soluciones: 
        # Mantener solo las que darían el mismo resultado si 'intento' fuera el secreto
        posibilidades = [
            p for p in posibilidades 
            if calcular_picas_y_fijas(intento, p) == (picas, fijas)
        ]
        
        # Tomar la primera opción disponible (Heurística simple y rápida)
        intento = posibilidades[0]


def estrategia_minimax(secreto, primer_intento="1234"):
    """Resuelve el juego optimizando el peor escenario (Minimax de Knuth)."""
    # Universo completo de combinaciones
    universo = ["".join(p) for p in itertools.permutations("0123456789", 4)]
    # Lista de soluciones que siguen siendo posibles candidatos
    posibilidades = list(universo)
    
    intento = primer_intento
    intentos_realizados = 0
    
    print(f"--- Iniciando Estrategia Minimax para el secreto: {secreto} ---")
    
    while True:
        intentos_realizados += 1
        picas, fijas = calcular_picas_y_fijas(intento, secreto)
        print(f"Intento {intentos_realizados}: {intento} -> Picas: {picas}, Fijas: {fijas}")
        
        if fijas == 4:
            print(f"🎉 ¡Ganó en {intentos_realizados} intentos!\n")
            return intentos_realizados
            
        # Podar las posibilidades actuales
        posibilidades = [
            p for p in posibilidades 
            if calcular_picas_y_fijas(intento, p) == (picas, fijas)
        ]
        
        # Si solo queda una opción, esa es la respuesta
        if len(posibilidades) == 1:
            intento = posibilidades[0]
            continue

        # --- ALGORITMO MINIMAX ---
        mejor_intento = None
        min_max_restantes = float('inf')
        
        # Optimización de tiempo: Si quedan pocos candidatos, 
        # buscar solo dentro de 'posibilidades' en lugar de todo el 'universo'.
        candidatos_a_evaluar = universo if len(posibilidades) > 20 else posibilidades

        for candidato in candidatos_a_evaluar:
            # Contar el tamaño de los grupos para cada respuesta posible (picas, fijas)
            conteos_respuestas = {}
            for pos in posibilidades:
                res = calcular_picas_y_fijas(candidato, pos)
                conteos_respuestas[res] = conteos_respuestas.get(res, 0) + 1
            
            # El peor escenario es el tamaño del grupo más grande
            peor_escenario = max(conteos_respuestas.values())
            
            # Queremos el candidato que minimice ese peor escenario
            if peor_escenario < min_max_restantes:
                min_max_restantes = peor_escenario
                mejor_intento = candidato
            # Romper empates: Preferir un candidato que pertenezca a las soluciones posibles
            elif peor_escenario == min_max_restantes and candidato in posibilidades and mejor_intento not in posibilidades:
                mejor_intento = candidato
                
        intento = mejor_intento

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
    
if __name__ == "__main__":
    # Definimos un número secreto difícil con dígitos únicos
    numero_secreto = generar_primer_intento()
    numero_primer_intento = generar_primer_intento()
    # Ejecución 1: Estrategia Rápida
    estrategia_simple(numero_secreto, primer_intento=numero_primer_intento)
    
    # Ejecución 2: Estrategia Minimax (Teoría de Juegos)
    estrategia_minimax(numero_secreto, primer_intento=numero_primer_intento)

# Es posible ahorrarse un intento, eliminando la ultima comprobacion, puesto que normalmente solo queda 1 numero
