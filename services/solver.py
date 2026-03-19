"""
Módulo de lógica matemática para la resolución de sistemas de ecuaciones lineales.

Implementa los siguientes métodos numéricos/algebraicos:
- Gauss-Jordan       : Eliminación Gaussiana con reducción total (RREF).
- Matriz Inversa     : Cálculo de la inversa mediante Gauss-Jordan.
- Reducción Gaussiana: Eliminación hacia adelante + sustitución regresiva.
- Método Gráfico    : Graficación de rectas e intersección para sistemas 2×2.
- Regla de Cramer   : Uso de determinantes para sistemas cuadrados (det ≠ 0).

Cada función devuelve un diccionario estandarizado con los pasos intermedios
y la solución, compatible con el modelo `SolutionResponse` del router.
"""

import copy


# ---------------------------------------------------------------------------
# Gauss-Jordan
# ---------------------------------------------------------------------------

def gauss_jordan(matrix, constants=None):
    """
    Resuelve un sistema de ecuaciones lineales por el método de Gauss-Jordan.

    El algoritmo transforma la matriz aumentada [A|b] en su forma escalonada
    reducida por filas (RREF). Al finalizar, cada fila pivot tiene un 1 en la
    diagonal y 0 en el resto de la columna, lo que permite leer la solución.
    Maneja sistemas con solución única, infinitas soluciones o sin solución.

    Args:
        matrix (list[list[float]]): Matriz de coeficientes A (n×m).
        constants (list[float] | None): Vector de términos independientes b.
            Si es None, opera solo sobre la matriz A (p. ej., para escalonar).

    Returns:
        dict: {
            "steps": list con cada transformación intermedias,
            "solution": list[float|str] con los valores de las variables o expresiones parametrizadas (o None).
            "solution_type": str ("Única", "Infinitas", "Sin solución"),
            "message": str
        }

    Raises:
        ValueError: Si el número de filas no coincide con el número de constantes.
    """
    if constants and len(matrix) != len(constants):
        raise ValueError("Las filas de la matriz deben coincidir con el número de constantes.")

    # Construir la matriz aumentada [A|b] o copiar A si no hay constantes
    if constants:
        aug = [row + [c] for row, c in zip(matrix, constants)]
    else:
        aug = copy.deepcopy(matrix)

    steps = []
    steps.append({"description": "Matriz Inicial", "matrix": copy.deepcopy(aug)})

    rows = len(aug)
    cols = len(aug[0])
    var_cols = cols - 1 if constants else cols

    r = 0
    pivots = []  # Para guardar (fila, columna) de cada pivot

    for c in range(var_cols):
        if r >= rows:
            break

        pivot = aug[r][c]

        # Si el pivot es 0, buscar una fila inferior con valor no nulo e intercambiar
        if pivot == 0:
            for j in range(r + 1, rows):
                if aug[j][c] != 0:
                    aug[r], aug[j] = aug[j], aug[r]
                    steps.append({"description": f"Intercambio F{r+1} y F{j+1}", "matrix": copy.deepcopy(aug)})
                    pivot = aug[r][c]
                    break
            
        if pivot == 0:
            continue  # Columna toda en ceros desde esta fila hacia abajo, pasar a la siguiente

        # Normalizar la fila pivot para que el pivot sea 1
        if pivot != 1:
            aug[r] = [val / pivot for val in aug[r]]
            steps.append({"description": f"F{r+1} / {pivot:.2f}", "matrix": copy.deepcopy(aug)})

        pivots.append((r, c))

        # Eliminar todos los demás elementos de la columna c (excepto el pivot)
        for k in range(rows):
            if k != r:
                factor = aug[k][c]
                if factor != 0:
                    aug[k] = [val_k - factor * val_r for val_k, val_r in zip(aug[k], aug[r])]
                    steps.append({"description": f"F{k+1} - ({factor:.2f})*F{r+1}", "matrix": copy.deepcopy(aug)})
        
        r += 1

    solution = None
    solution_type = "Única"
    message = "El sistema tiene una solución única."

    if constants:
        has_no_sol = False
        for i in range(rows):
            es_cero = all(abs(aug[i][j]) < 1e-9 for j in range(var_cols))
            indep = aug[i][-1]
            if es_cero and abs(indep) > 1e-9:
                has_no_sol = True
                break

        if has_no_sol:
            solution_type = "Sin solución"
            message = "El sistema no tiene solución (es incompatible)."
            solution = None
        else:
            pivot_cols = [p[1] for p in pivots]
            free_vars = [c for c in range(var_cols) if c not in pivot_cols]

            if len(free_vars) > 0:
                solution_type = "Infinitas"
                message = "El sistema tiene infinitas soluciones. Ecuaciones parametrizadas:"
                solution_dict = {}
                param_letters = ['t', 's', 'r', 'u', 'v', 'w']

                for idx, c_idx in enumerate(free_vars):
                    param = param_letters[idx % len(param_letters)] if idx < len(param_letters) else f"t{idx+1}"
                    solution_dict[f"x{c_idx+1}"] = param

                for r_idx, c_idx in pivots:
                    const_val = aug[r_idx][-1]
                    expr_parts = []

                    if abs(const_val) > 1e-9:
                        expr_parts.append(f"{round(const_val, 4)}")

                    for free_c in free_vars:
                        coef = aug[r_idx][free_c]
                        if abs(coef) > 1e-9:
                            sign = "+" if -coef > 0 else "-"
                            val = abs(-coef)
                            str_val = "" if abs(val - 1.0) < 1e-9 else f"{round(val, 4)}"
                            param_name = solution_dict[f"x{free_c+1}"]
                            expr_parts.append(f"{sign} {str_val}{param_name}")

                    if not expr_parts:
                        solution_dict[f"x{c_idx+1}"] = "0"
                    else:
                        expr = " ".join(expr_parts).strip()
                        if expr.startswith("+ "):
                            expr = expr[2:]
                        solution_dict[f"x{c_idx+1}"] = expr

                solution = [solution_dict.get(f"x{i+1}", f"x{i+1}") for i in range(var_cols)]
            else:
                solution_type = "Única"
                message = "El sistema tiene una solución única."
                solution = [0.0] * var_cols
                for r_idx, c_idx in pivots:
                    solution[c_idx] = round(aug[r_idx][-1], 4)

    return {
        "steps": steps, 
        "solution": solution, 
        "solution_type": solution_type,
        "message": message
    }


# ---------------------------------------------------------------------------
# Matriz Inversa
# ---------------------------------------------------------------------------

def inverse_matrix(matrix, constants=None):
    """
    Calcula la inversa de una matriz cuadrada y, opcionalmente, resuelve Ax = b.

    El algoritmo construye la matriz ampliada [A|I] (donde I es la identidad
    del mismo tamaño) y le aplica Gauss-Jordan. Cuando la parte izquierda
    llega a ser la identidad, la parte derecha es A⁻¹.

    Si se proporcionan constantes b, la solución x se calcula como x = A⁻¹ · b.

    Args:
        matrix (list[list[float]]): Matriz cuadrada de coeficientes A (n×n).
        constants (list[float] | None): Vector b. Si se proporciona, se
            calcula también la solución x = A⁻¹ · b.

    Returns:
        dict: {
            "steps": pasos intermedios incluyendo la inversa resultante,
            "solution": list[float] con los valores de x (o None),
            "inverse": list[list[float]] con la matriz inversa A⁻¹.
        }

    Raises:
        ValueError: Si la matriz no es cuadrada o si es singular (det = 0).
    """
    rows = len(matrix)
    cols = len(matrix[0])

    if rows != cols:
        raise ValueError("La matriz debe ser cuadrada")

    # Construir [A|I]: ampliar cada fila con la fila correspondiente de la identidad
    aug = [row + [1.0 if i == j else 0.0 for j in range(rows)] for i, row in enumerate(matrix)]
    steps = [{"description": "Matriz ampliada con la Identidad", "matrix": copy.deepcopy(aug)}]

    for i in range(rows):
        pivot = aug[i][i]

        # Buscar fila intercambiable si el pivot actual es cero
        if pivot == 0:
            for j in range(i + 1, rows):
                if aug[j][i] != 0:
                    aug[i], aug[j] = aug[j], aug[i]
                    steps.append({"description": f"Intercambio F{i+1} y F{j+1}", "matrix": copy.deepcopy(aug)})
                    pivot = aug[i][i]
                    break
            if pivot == 0:
                raise ValueError("La matriz no es invertible (determinante 0)")

        # Normalizar la fila pivot
        if pivot != 1:
            aug[i] = [val / pivot for val in aug[i]]
            steps.append({"description": f"F{i+1} / {pivot:.2f}", "matrix": copy.deepcopy(aug)})

        # Eliminar todos los demás elementos de la columna
        for k in range(rows):
            if k != i:
                factor = aug[k][i]
                if factor != 0:
                    aug[k] = [val_k - factor * val_i for val_k, val_i in zip(aug[k], aug[i])]
                    steps.append({"description": f"F{k+1} - ({factor:.2f})*F{i+1}", "matrix": copy.deepcopy(aug)})

    # Extraer la parte derecha de la matriz ampliada: esa es A⁻¹
    inverse = [[round(val, 4) for val in row[rows:]] for row in aug]
    steps.append({"description": "Matriz inversa resultante", "matrix": copy.deepcopy(inverse)})

    # Calcular la solución x = A⁻¹ · b si se proporcionaron constantes
    solution = None
    if constants:
        if len(constants) != rows:
            raise ValueError("Discrepancia en las dimensiones de las constantes")
        solution = [
            round(sum(inverse[i][j] * constants[j] for j in range(rows)), 4)
            for i in range(rows)
        ]
        steps.append({"description": "Solución multiplicando inversa por constantes", "matrix": [[s] for s in solution]})

    return {"steps": steps, "solution": solution, "inverse": inverse}


# ---------------------------------------------------------------------------
# Reducción Gaussiana (Eliminación hacia adelante + sustitución regresiva)
# ---------------------------------------------------------------------------

def reduction(matrix, constants=None):
    """
    Resuelve un sistema de ecuaciones lineales mediante Reducción Gaussiana.

    A diferencia de Gauss-Jordan, este método solo lleva la matriz a la
    forma escalonada superior (triangular), sin eliminar los elementos
    por encima de los pivots. La solución se obtiene mediante sustitución
    regresiva (back substitution).

    Pasos del algoritmo:
        1. Construir la matriz aumentada [A|b].
        2. Para cada pivot, anular todos los elementos debajo de él.
        3. Aplicar sustitución regresiva para obtener x.

    Args:
        matrix (list[list[float]]): Matriz de coeficientes A (n×m).
        constants (list[float] | None): Vector de términos independientes b.

    Returns:
        dict: {
            "steps": pasos de la eliminación,
            "solution": list[float] con los valores de las variables (o None).
        }

    Raises:
        ValueError: Si el sistema es incompatible (sin solución).
    """
    if constants:
        aug = [row + [c] for row, c in zip(matrix, constants)]
    else:
        aug = copy.deepcopy(matrix)

    steps = [{"description": "Matriz Inicial (Reducción Gaussiana)", "matrix": copy.deepcopy(aug)}]
    rows = len(aug)
    cols = len(aug[0])

    # Eliminación hacia adelante: construir la forma triangular superior
    for i in range(min(rows, cols - (1 if constants else 0))):
        pivot = aug[i][i]

        # Buscar intercambio si el pivot actual es cero
        if pivot == 0:
            for j in range(i + 1, rows):
                if aug[j][i] != 0:
                    aug[i], aug[j] = aug[j], aug[i]
                    steps.append({"description": f"Intercambio F{i+1} y F{j+1}", "matrix": copy.deepcopy(aug)})
                    pivot = aug[i][i]
                    break
            if pivot == 0:
                continue  # Columna toda en ceros

        # Anular únicamente los elementos debajo del pivot
        for k in range(i + 1, rows):
            factor = aug[k][i] / pivot
            if factor != 0:
                aug[k] = [val_k - factor * val_i for val_k, val_i in zip(aug[k], aug[i])]
                steps.append({"description": f"F{k+1} - ({factor:.2f})*F{i+1}", "matrix": copy.deepcopy(aug)})

    # Sustitución regresiva para hallar los valores de las variables
    solution = None
    if constants:
        ans = [0.0] * rows
        for i in range(rows - 1, -1, -1):
            # Valor acumulado de los términos ya resueltos (variables siguientes)
            val = aug[i][-1] - sum(aug[i][j] * ans[j] for j in range(i + 1, rows))
            if aug[i][i] == 0:
                if abs(val) > 1e-9:
                    raise ValueError("Sistema Incompatible")
            else:
                ans[i] = val / aug[i][i]
        solution = [round(x, 4) for x in ans]

    return {"steps": steps, "solution": solution}


# ---------------------------------------------------------------------------
# Método Gráfico
# ---------------------------------------------------------------------------

def graphical(matrix, constants=None):
    """
    Resuelve gráficamente un sistema de 2 ecuaciones lineales con 2 incógnitas.

    Calcula dos puntos representativos de cada recta (en x = -10 y x = 10)
    y determina la intersección como solución del sistema usando la fórmula
    directa de Cramer para sistemas 2×2.

    Formato esperado del sistema:
        a₁x + b₁y = c₁
        a₂x + b₂y = c₂

    Args:
        matrix (list[list[float]]): Matriz 2×2 de coeficientes [[a₁,b₁],[a₂,b₂]].
        constants (list[float]): Vector de constantes [c₁, c₂]. Obligatorio.

    Returns:
        dict: {
            "steps": pasos del cálculo,
            "solution": [x, y] punto de intersección (o None si no es único),
            "lines": lista de rectas con ecuación legible y puntos representativos,
            "error": mensaje si las rectas son paralelas o coincidentes.
        }

    Raises:
        ValueError: Si la entrada no es un sistema 2×2 con 2 constantes.
    """
    if not constants or len(matrix) != 2 or len(matrix[0]) != 2 or len(constants) != 2:
        raise ValueError("El método gráfico requiere un sistema de 2 ecuaciones con 2 variables")

    steps = [{"description": "Ecuaciones detectadas", "matrix": copy.deepcopy(matrix)}]

    lines = []
    for i, (row, c) in enumerate(zip(matrix, constants)):
        pts = []
        a, b = row[0], row[1]

        if b != 0:
            # Calcular y para x = -10 y x = 10 despejando: y = (c - a·x) / b
            y1 = (c - a * (-10)) / b
            y2 = (c - a * (10)) / b
            pts = [{"x": -10, "y": round(y1, 2)}, {"x": 10, "y": round(y2, 2)}]
        else:
            # Recta vertical: x = c/a
            if a == 0:
                continue  # Ecuación degenerada (0x + 0y = c)
            x = c / a
            pts = [{"x": round(x, 2), "y": -10}, {"x": round(x, 2), "y": 10}]

        # Construir la cadena legible de la ecuación, p. ej. "2x + y = 4"
        def fmt_coeff(val, var, is_first):
            """Formatea un coeficiente como texto para la representación de la ecuación."""
            if val == 0:
                return ""
            abs_v = abs(val)
            num = int(abs_v) if abs_v == int(abs_v) else round(abs_v, 2)
            coeff_str = "" if num == 1 else str(num)
            term = f"{coeff_str}{var}"
            if is_first:
                return f"-{term}" if val < 0 else term
            return f" - {term}" if val < 0 else f" + {term}"

        eq_a = fmt_coeff(a, "x", True)
        eq_b_str = fmt_coeff(b, "y", eq_a == "")
        c_disp = int(c) if c == int(c) else round(c, 2)
        eq_str = f"{eq_a}{eq_b_str} = {c_disp}"

        lines.append({"equation": eq_str, "points": pts})
        steps.append({"description": f"Puntos calculados para Ec {i+1}", "points": pts})

    # Calcular la intersección usando la fórmula de Cramer para 2×2
    det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    solution = None
    error = None

    if det != 0:
        x = (constants[0] * matrix[1][1] - constants[1] * matrix[0][1]) / det
        y = (matrix[0][0] * constants[1] - matrix[1][0] * constants[0]) / det
        solution = [round(x, 4), round(y, 4)]
        steps.append({"description": "Intersección encontrada", "solution": solution})
    else:
        error = "Las líneas son paralelas (sin solución) o coincidentes (infinitas soluciones)"

    return {"steps": steps, "solution": solution, "lines": lines, "error": error}


# ---------------------------------------------------------------------------
# Utilidad: Cálculo del determinante (recursivo por expansión de cofactores)
# ---------------------------------------------------------------------------

def get_determinant(matrix):
    """
    Calcula el determinante de una matriz cuadrada mediante expansión de cofactores.

    Para matrices 1×1 y 2×2 usa fórmulas directas. Para matrices mayores
    aplica recursión expandiendo por la primera fila.

    Args:
        matrix (list[list[float]]): Matriz cuadrada de cualquier tamaño.

    Returns:
        float: Valor del determinante de la matriz.
    """
    # Caso base: matriz 1×1
    if len(matrix) == 1:
        return matrix[0][0]

    # Caso base: matriz 2×2
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Caso general: expansión por cofactores a lo largo de la primera fila
    det = 0
    for c in range(len(matrix)):
        # Submatriz que excluye la primera fila y la columna c
        submatrix = [row[:c] + row[c + 1:] for row in matrix[1:]]
        sign = (-1) ** c  # Signo del cofactor: + para columnas pares, - para impares
        det += sign * matrix[0][c] * get_determinant(submatrix)
    return det


# ---------------------------------------------------------------------------
# Regla de Cramer
# ---------------------------------------------------------------------------

def cramer(matrix, constants=None):
    """
    Resuelve un sistema de ecuaciones lineales usando la Regla de Cramer.

    La Regla de Cramer calcula la solución de Ax = b calculando:
        xᵢ = det(Aᵢ) / det(A)
    donde Aᵢ es la matriz A con la columna i reemplazada por el vector b.

    Requisitos:
        - El sistema debe ser cuadrado (n ecuaciones, n incógnitas).
        - El determinante principal det(A) debe ser ≠ 0 (sistema compatible determinado).

    Args:
        matrix (list[list[float]]): Matriz cuadrada de coeficientes A (n×n).
        constants (list[float]): Vector de términos independientes b. Obligatorio.

    Returns:
        dict: {
            "steps": pasos mostrando cada determinante calculado,
            "solution": list[float] con los valores de las variables x₁, x₂, ..., xₙ.
        }

    Raises:
        ValueError: Si no se proporcionan constantes, la matriz no es cuadrada,
            las dimensiones no coinciden, o el determinante es cero.
    """
    if not constants:
        raise ValueError("La regla de Cramer requiere constantes (términos independientes).")

    rows = len(matrix)
    cols = len(matrix[0])

    if rows != cols:
        raise ValueError(
            "La regla de Cramer requiere una matriz cuadrada "
            "(mismo número de ecuaciones que de variables)."
        )

    if len(constants) != rows:
        raise ValueError("El número de constantes debe coincidir con el número de ecuaciones.")

    steps = [{"description": "Matriz del sistema", "matrix": copy.deepcopy(matrix)}]

    # Calcular el determinante principal Δ
    det_sys = get_determinant(matrix)
    steps.append({"description": f"Determinante del sistema (Δ) = {round(det_sys, 4)}"})

    if abs(det_sys) < 1e-9:
        raise ValueError(
            "El determinante del sistema es 0. "
            "La regla de Cramer no es aplicable (el sistema no tiene solución única)."
        )

    solution = []

    # Para cada variable xᵢ: reemplazar la columna i con las constantes y calcular det
    for i in range(cols):
        mod_matrix = [row[:] for row in matrix]
        for r in range(rows):
            mod_matrix[r][i] = constants[r]

        steps.append({
            "description": f"Matriz para la variable x{i+1} (Reemplazando columna {i+1} por constantes)",
            "matrix": mod_matrix,
        })

        det_var = get_determinant(mod_matrix)
        val = det_var / det_sys  # xᵢ = Δxᵢ / Δ
        solution.append(round(val, 4))

        steps.append({
            "description": (
                f"Determinante (Δx{i+1}) = {round(det_var, 4)} -> "
                f"x{i+1} = {round(det_var, 4)} / {round(det_sys, 4)} = {round(val, 4)}"
            )
        })

    return {"steps": steps, "solution": solution}
