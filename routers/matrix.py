"""
Router de resolución de sistemas de ecuaciones lineales.

Este módulo expone los endpoints REST bajo el prefijo `/solve`.
Cada endpoint recibe una matriz y, opcionalmente, un vector de constantes,
y devuelve los pasos intermedios del método seleccionado junto con la solución.

Métodos disponibles:
- Gauss-Jordan      : /solve/gauss-jordan
- Matriz inversa    : /solve/inverse
- Reducción Gaussiana: /solve/reduction
- Método gráfico   : /solve/graphical
- Regla de Cramer  : /solve/cramer
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
from services import solver

# El prefijo agrupa todos los endpoints de resolución bajo /solve
router = APIRouter(prefix="/solve", tags=["Resolución de Matrices"])


# ---------------------------------------------------------------------------
# Modelos de datos (esquemas Pydantic)
# ---------------------------------------------------------------------------

class MatrixInput(BaseModel):
    """
    Cuerpo de la solicitud para todos los métodos de resolución.

    Atributos:
        matrix (List[List[float]]): Matriz de coeficientes del sistema.
            Cada sublista representa una fila.
            Ejemplo: [[2, 1], [5, 3]]
        constants (Optional[List[float]]): Vector de términos independientes
            (lado derecho de las ecuaciones). Requerido en la mayoría de
            los métodos excepto cuando se calcula solo la inversa o la forma
            escalonada sin solución.
            Ejemplo: [4, 7]
    """
    matrix: List[List[float]] = Field(
        ...,
        example=[[2, 1], [5, 3]],
        description="Matriz de coeficientes del sistema de ecuaciones.",
    )
    constants: Optional[List[float]] = Field(
        default=None,
        example=[4, 7],
        description="Vector de términos independientes (opcional según el método).",
    )


class SolutionResponse(BaseModel):
    """
    Respuesta estándar devuelta por todos los endpoints de resolución.

    Atributos:
        steps (List[Dict]): Lista de pasos intermedios del método aplicado.
            Cada paso incluye una descripción textual y el estado de la
            matriz en ese momento.
        solution (Optional[Any]): Vector solución del sistema (valores de
            las variables). Es None si no existe solución única.
        inverse (Optional[Any]): Matriz inversa calculada. Solo presente en
            el método de la matriz inversa.
        lines (Optional[List[Dict]]): Puntos y ecuaciones de las rectas.
            Solo presente en el método gráfico.
        error (Optional[str]): Mensaje de error descriptivo cuando el sistema
            no tiene solución única (rectas paralelas, determinante cero, etc.).
        solution_type (Optional[str]): Indica si la solución es Única, Infinitas o Sin solución.
        message (Optional[str]): Mensaje legible sobre el tipo de solución.
    """
    steps: List[Dict[str, Any]] = Field(description="Pasos intermedios del método.")
    solution: Optional[Any] = Field(default=None, description="Solución del sistema.")
    inverse: Optional[Any] = Field(default=None, description="Matriz inversa (solo método inversa).")
    lines: Optional[List[Dict[str, Any]]] = Field(default=None, description="Rectas graficables (solo método gráfico).")
    error: Optional[str] = Field(default=None, description="Mensaje de error si no hay solución única.")
    solution_type: Optional[str] = Field(default=None, description="Tipo de solución (Única, Infinitas, Sin solución).")
    message: Optional[str] = Field(default=None, description="Mensaje explicativo de la solución.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/gauss-jordan",
    response_model=SolutionResponse,
    summary="Resolver por Gauss-Jordan",
)
def solve_gauss_jordan(data: MatrixInput):
    """
    Resuelve el sistema de ecuaciones lineales usando el método de **Gauss-Jordan**.

    El método transforma la matriz aumentada [A|b] en la forma escalonada
    reducida por filas (RREF), obteniendo directamente la solución sin
    necesidad de sustitución regresiva.

    - **Entrada**: Matriz de coeficientes + vector de constantes (recomendado).
    - **Salida**: Pasos de eliminación y solución del sistema.

    Lanza HTTP 400 si la entrada es inválida o el sistema no tiene solución única.
    """
    try:
        return solver.gauss_jordan(data.matrix, data.constants)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/inverse",
    response_model=SolutionResponse,
    summary="Resolver por Matriz Inversa",
)
def solve_inverse(data: MatrixInput):
    """
    Resuelve el sistema mediante el cálculo de la **matriz inversa**.

    Calcula A⁻¹ usando eliminación Gauss-Jordan sobre la matriz ampliada
    [A|I]. Si se proporcionan constantes, también calcula x = A⁻¹ · b.

    - **Requisito**: La matriz debe ser cuadrada y no singular (det ≠ 0).
    - **Salida**: Pasos de cálculo, la inversa A⁻¹ y la solución (si hay constantes).

    Lanza HTTP 400 si la matriz es singular o no cuadrada.
    """
    try:
        return solver.inverse_matrix(data.matrix, data.constants)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/reduction",
    response_model=SolutionResponse,
    summary="Resolver por Reducción Gaussiana",
)
def solve_reduction(data: MatrixInput):
    """
    Resuelve el sistema mediante **Reducción Gaussiana** (eliminación hacia adelante).

    A diferencia de Gauss-Jordan, este método solo lleva la matriz a la
    forma escalonada superior y luego aplica sustitución regresiva para
    obtener la solución.

    - **Entrada**: Matriz de coeficientes + vector de constantes.
    - **Salida**: Pasos de reducción y solución del sistema.

    Lanza HTTP 400 si el sistema es incompatible (sin solución).
    """
    try:
        return solver.reduction(data.matrix, data.constants)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/graphical",
    response_model=SolutionResponse,
    summary="Resolver por Método Gráfico",
)
def solve_graphical(data: MatrixInput):
    """
    Resuelve un sistema de **2 ecuaciones con 2 incógnitas** de forma gráfica.

    Calcula dos puntos para cada recta y determina su intersección
    (solución del sistema) usando la fórmula de Cramer para 2×2.

    - **Requisito estricto**: La matriz debe ser exactamente 2×2 con 2 constantes.
    - **Salida**: Puntos de cada recta, la cadena de la ecuación legible
      y la intersección como solución.

    Lanza HTTP 400 si las rectas son paralelas o si la entrada no es 2×2.
    """
    try:
        return solver.graphical(data.matrix, data.constants)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/cramer",
    response_model=SolutionResponse,
    summary="Resolver por Regla de Cramer",
)
def solve_cramer(data: MatrixInput):
    """
    Resuelve el sistema de ecuaciones usando la **Regla de Cramer**.

    Calcula el determinante principal Δ y un determinante Δxᵢ por cada
    variable, reemplazando la columna correspondiente por las constantes.
    La solución es xᵢ = Δxᵢ / Δ.

    - **Requisito**: La matriz debe ser cuadrada y el determinante distinto
      de cero (sistema con solución única).
    - **Salida**: Determinantes calculados en cada paso y la solución final.

    Lanza HTTP 400 si el determinante es cero o la entrada es inválida.
    """
    try:
        return solver.cramer(data.matrix, data.constants)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
