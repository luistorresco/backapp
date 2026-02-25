from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from services import solver

router = APIRouter(prefix="/solve", tags=["matrix"])

class MatrixInput(BaseModel):
    matrix: List[List[float]]
    constants: Optional[List[float]] = None

class SolutionResponse(BaseModel):
    steps: List[Dict[str, Any]]
    solution: Optional[Any] = None
    inverse: Optional[Any] = None
    lines: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None

@router.post("/gauss-jordan", response_model=SolutionResponse)
def solve_gauss_jordan(data: MatrixInput):
    try:
        return solver.gauss_jordan(data.matrix, data.constants)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/inverse", response_model=SolutionResponse)
def solve_inverse(data: MatrixInput):
    try:
        return solver.inverse_matrix(data.matrix, data.constants)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/reduction", response_model=SolutionResponse)
def solve_reduction(data: MatrixInput):
    try:
        return solver.reduction(data.matrix, data.constants)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/graphical", response_model=SolutionResponse)
def solve_graphical(data: MatrixInput):
    try:
        return solver.graphical(data.matrix, data.constants)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
