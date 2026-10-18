# Pasos para ejecutar el agente

## Iniciar el navegador en modo debug
& "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="C:\Users\juanp\chrome-agente"

## Iniciar el juego en el navegador
https://www.minijuegos.com/embed/diamond-rush#

## Inicio y Manejo del Agente

### Modo Normal
Este comando es para ejecutar el agente en el modo normal, es decir cada diez movimiento el agente toma un nuevo pantallazo para verficar si su estado y realizar correcciones.

python main.py

### Modo Alternativo
Este comando permite que el agente realice un pantallazo despues de cada movimiento si la bandera es 1 o ejecutar todo el plan propuesto y verificar el plan al final si la bandera es 0

python main.py --verify-every 0

python main.py --verify-every 1


## Niveles con mas de un boton
Ahora, para el caso donde los niveles tienen mas de un boton, hay que especificarle al agente el nivel (Esto es para que el agente, pueda leer un archivo interno con el mapeo de los botones y en base a eso arme el camino)

Esto se puede lograr con la bandera --level N, donde N es el nivel en que se encuentra el agente

Por ejemplo

python main.py --verify-every 0 --level 8
python main.py --level 8