import numpy as np

def lsqr(A, b, iters=10):
    m, n = A.shape
    x = np.zeros(n)
    
    # Шаг 1: Инициализация
    beta = np.linalg.norm(b)
    u = b / (1)____
    alpha = np.linalg.norm(A.T @ u)
    v = (A.T @ u) / (2)___
    
    w = v.copy()
    phi_bar = beta
    rho_bar = alpha
    
    history = [x.copy()]
    
    for i in range(iters):
        # 2. Бидиагонализация Голуба-Кахана
        u = A @ v - alpha * (3)____
        beta = np.linalg.norm(u)
        u /= beta
        
        v = A.T @ u - beta * (4)____
        alpha = np.linalg.norm(v)
        v /= alpha
        
        # 3. Вращения Гивенса
        rho = np.sqrt(rho_bar**2 + beta**2)
        c = rho_bar / rho
        s = beta / rho
        
        theta = s * alpha
        rho_bar = -c * alpha
        phi = c * phi_bar
        phi_bar = s * phi_bar
        
        # 4. Обновление решения
        x = x + (phi / rho) * (5)___
        w = v - (theta / rho) * w
        
        history.append(x.copy())
        
        # Проверка сходимости по невязке (упрощенно)
        if abs(phi_bar) < 1e-10:
            break
            
    return x, np.array(history)

# Пример: переопределенная система (3 уравнения, 2 неизвестных)
A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
b = np.array([1, 1, 1], dtype=float)

sol, hist = lsqr(A, b)
print(f"Решение LSQR: {sol}")
