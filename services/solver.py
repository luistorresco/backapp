import copy

def gauss_jordan(matrix, constants=None):
    if constants and len(matrix) != len(constants):
        raise ValueError("Matrix rows must match number of constants.")

    if constants:
        aug = [row + [c] for row, c in zip(matrix, constants)]
    else:
        aug = copy.deepcopy(matrix)
    
    steps = []
    steps.append({"description": "Matriz Inicial", "matrix": copy.deepcopy(aug)})
    
    rows = len(aug)
    cols = len(aug[0])
    
    for i in range(rows):
        if i >= cols - (1 if constants else 0):
            break
            
        pivot = aug[i][i]
        if pivot == 0:
            for j in range(i+1, rows):
                if aug[j][i] != 0:
                    aug[i], aug[j] = aug[j], aug[i]
                    steps.append({"description": f"Intercambio F{i+1} y F{j+1}", "matrix": copy.deepcopy(aug)})
                    pivot = aug[i][i]
                    break
            if pivot == 0:
                continue
        
        if pivot != 1:
            aug[i] = [val / pivot for val in aug[i]]
            steps.append({"description": f"F{i+1} / {pivot:.2f}", "matrix": copy.deepcopy(aug)})
            
        for k in range(rows):
            if k != i:
                factor = aug[k][i]
                if factor != 0:
                    aug[k] = [val_k - factor * val_i for val_k, val_i in zip(aug[k], aug[i])]
                    steps.append({"description": f"F{k+1} - ({factor:.2f})*F{i+1}", "matrix": copy.deepcopy(aug)})
    
    solution = None
    if constants:
        solution = [round(row[-1], 4) for row in aug]
        
    return {"steps": steps, "solution": solution}

def inverse_matrix(matrix, constants=None):
    rows = len(matrix)
    cols = len(matrix[0])
    if rows != cols:
        raise ValueError("La matriz debe ser cuadrada")
        
    aug = [row + [1.0 if i == j else 0.0 for j in range(rows)] for i, row in enumerate(matrix)]
    steps = [{"description": "Matriz ampliada con la Identidad", "matrix": copy.deepcopy(aug)}]
    
    for i in range(rows):
        pivot = aug[i][i]
        if pivot == 0:
            for j in range(i+1, rows):
                if aug[j][i] != 0:
                    aug[i], aug[j] = aug[j], aug[i]
                    steps.append({"description": f"Intercambio F{i+1} y F{j+1}", "matrix": copy.deepcopy(aug)})
                    pivot = aug[i][i]
                    break
            if pivot == 0:
                raise ValueError("La matriz no es invertible (determinante 0)")
                
        if pivot != 1:
            aug[i] = [val / pivot for val in aug[i]]
            steps.append({"description": f"F{i+1} / {pivot:.2f}", "matrix": copy.deepcopy(aug)})
            
        for k in range(rows):
            if k != i:
                factor = aug[k][i]
                if factor != 0:
                    aug[k] = [val_k - factor * val_i for val_k, val_i in zip(aug[k], aug[i])]
                    steps.append({"description": f"F{k+1} - ({factor:.2f})*F{i+1}", "matrix": copy.deepcopy(aug)})
    
    inverse = [[round(val, 4) for val in row[rows:]] for row in aug]
    steps.append({"description": "Matriz inversa resultante", "matrix": copy.deepcopy(inverse)})
    
    solution = None
    if constants:
        if len(constants) != rows:
            raise ValueError("Constants dimension mismatch")
        solution = [round(sum(inverse[i][j] * constants[j] for j in range(rows)), 4) for i in range(rows)]
        steps.append({"description": "Solución multiplicando inversa por constantes", "matrix": [[s] for s in solution]})
        
    return {"steps": steps, "solution": solution, "inverse": inverse}

def reduction(matrix, constants=None):
    if constants:
        aug = [row + [c] for row, c in zip(matrix, constants)]
    else:
        aug = copy.deepcopy(matrix)
        
    steps = [{"description": "Matriz Inicial (Reducción Gaussiana)", "matrix": copy.deepcopy(aug)}]
    rows = len(aug)
    cols = len(aug[0])
    
    for i in range(min(rows, cols - (1 if constants else 0))):
        pivot = aug[i][i]
        if pivot == 0:
            for j in range(i+1, rows):
                if aug[j][i] != 0:
                    aug[i], aug[j] = aug[j], aug[i]
                    steps.append({"description": f"Intercambio F{i+1} y F{j+1}", "matrix": copy.deepcopy(aug)})
                    pivot = aug[i][i]
                    break
            if pivot == 0:
                continue
        
        for k in range(i+1, rows):
            factor = aug[k][i] / pivot
            if factor != 0:
                aug[k] = [val_k - factor * val_i for val_k, val_i in zip(aug[k], aug[i])]
                steps.append({"description": f"F{k+1} - ({factor:.2f})*F{i+1}", "matrix": copy.deepcopy(aug)})
                
    solution = None
    if constants:
        ans = [0.0] * rows
        for i in range(rows-1, -1, -1):
            val = aug[i][-1] - sum(aug[i][j] * ans[j] for j in range(i+1, rows))
            if aug[i][i] == 0:
                if abs(val) > 1e-9:
                    raise ValueError("Sistema Incompatible")
            else:
                ans[i] = val / aug[i][i]
        solution = [round(x, 4) for x in ans]
        
    return {"steps": steps, "solution": solution}

def graphical(matrix, constants=None):
    if not constants or len(matrix) != 2 or len(matrix[0]) != 2 or len(constants) != 2:
        raise ValueError("El método gráfico requiere un sistema de 2 ecuaciones con 2 variables")
        
    steps = [{"description": "Ecuaciones detectadas", "matrix": copy.deepcopy(matrix)}]
    
    lines = []
    for i, (row, c) in enumerate(zip(matrix, constants)):
        pts = []
        a, b = row[0], row[1]
        
        if b != 0:
            y1 = (c - a*(-10)) / b
            y2 = (c - a*(10)) / b
            pts = [{"x": -10, "y": round(y1, 2)}, {"x": 10, "y": round(y2, 2)}]
        else:
            if a == 0:
                continue # 0x + 0y = c makes no sense for a line if c!=0 or creates entire plane
            x = c / a
            pts = [{"x": round(x, 2), "y": -10}, {"x": round(x, 2), "y": 10}]
            
        lines.append({"equation": f"Eq {i+1}", "points": pts})
        steps.append({"description": f"Puntos calculados para Eq {i+1}", "points": pts})
        
    det = matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
    solution = None
    if det != 0:
        x = (constants[0]*matrix[1][1] - constants[1]*matrix[0][1]) / det
        y = (matrix[0][0]*constants[1] - matrix[1][0]*constants[0]) / det
        solution = [round(x, 4), round(y, 4)]
        steps.append({"description": "Intersección encontrada", "solution": solution})
    else:
        raise ValueError("Las líneas son paralelas (sin solución) o coincidentes (infinitas soluciones)")
        
    return {"steps": steps, "solution": solution, "lines": lines}
