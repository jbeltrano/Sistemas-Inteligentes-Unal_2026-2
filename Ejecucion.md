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