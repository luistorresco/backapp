# Matrix Solver Backend API

Este es el backend para la aplicación de cálculo matricial, desarrollado con **FastAPI** y **Python**.

## Tecnologías Utilizadas

- **FastAPI**: Framework web rápido y moderno para la construcción de la API.
- **Uvicorn**: Servidor ASGI para ejecutar la aplicación.
- **Pydantic**: Para validación de datos de entrada/salida.
- **NumPy**: (Opcional) para operaciones matriciales avanzadas si es requerido.

## Requisitos

- Python 3.11 o superior.
- Docker y Docker Compose (Opcional, para contenerizar la aplicación).

## Instalación y Ejecución Local

1. Clona el repositorio y navega a la carpeta del backend.
2. Crea un entorno virtual e instálalo:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\\Scripts\\activate
   pip install -r requirements.txt
   ```
3. Ejecuta el servidor:
   ```bash
   uvicorn main:app --reload
   ```
4. La API estará disponible en `http://localhost:8000`. Puedes acceder a la documentación interactiva (Swagger UI) en `http://localhost:8000/docs`.

## Ejecución usando Docker

El backend incluye un `Dockerfile` y un `docker-compose.yml` para facilitar el despliegue.

Para levantarlo con Docker Compose:

```bash
docker-compose up --build
```
La API también estará disponible en `http://localhost:8000`.

## Estructura del Proyecto

- `main.py`: Punto de entrada de la aplicación y configuración de FastAPI y CORS.
- `routers/`: Contiene los controladores de las rutas (endpoints) (ej. `matrix.py`).
- `services/`: Contiene la lógica matemática para resolver los sistemas de ecuaciones (ej. `solver.py` con métodos como Gauss-Jordan, Cramer, etc.).
- `requirements.txt`: Dependencias de Python del proyecto.

## Endpoints Principales

- `GET /`: Mensaje de bienvenida de la API.
- Endpoints de cálculo en `/solve/` que toman como entrada un objeto `MatrixInput` que incluye la matriz principal y (opcionalmente) los términos independientes. (Admite varios métodos como Cramer, Gauss, Inversa, Graficación, etc.).
