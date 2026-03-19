"""
Punto de entrada principal de la API Juliga.

Este archivo inicializa la aplicación FastAPI, configura el middleware
de CORS para permitir solicitudes del frontend (aplicación Android),
y registra los routers de los distintos módulos.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import matrix

# ---------------------------------------------------------------------------
# Inicialización de la aplicación FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Juliga API",
    description=(
        "API REST para la resolución de sistemas de ecuaciones lineales. "
        "Soporta los métodos: Gauss-Jordan, Inversa, Reducción Gaussiana, "
        "Gráfico y Regla de Cramer. Desarrollada con FastAPI y Python."
    ),
    version="3.1.2",
    contact={
        "name": "Equipo Juliga",
    },
    license_info={
        "name": "MIT",
    },
)

# ---------------------------------------------------------------------------
# Configuración de CORS
# Permite que el frontend (app Android u otros clientes) realice peticiones
# a esta API sin restricciones de origen durante el desarrollo.
# En producción se recomienda especificar los orígenes permitidos.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Orígenes permitidos (todos en desarrollo)
    allow_credentials=False,   # Deshabilitado para permitir "*" en allow_origins
    allow_methods=["*"],       # Métodos HTTP permitidos (GET, POST, etc.)
    allow_headers=["*"],       # Cabeceras HTTP permitidas
)

# ---------------------------------------------------------------------------
# Registro de routers
# Cada router agrupa los endpoints de un dominio específico.
# ---------------------------------------------------------------------------
app.include_router(matrix.router)


# ---------------------------------------------------------------------------
# Endpoint raíz
# ---------------------------------------------------------------------------
@app.get("/", summary="Verificación de estado", tags=["General"])
def read_root():
    """
    Endpoint de bienvenida y verificación de estado de la API.

    Retorna un mensaje confirmando que el servidor está en línea.
    Útil para pruebas de conectividad (health-check).
    """
    return {"mensaje": "Bienvenido a Juliga API", "estado": "en línea"}
