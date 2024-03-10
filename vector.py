import numpy as np

def create_matrix(size):
    # Ask the user to input the elements of the matrix
    print(f"Enter {size**2} elements for a {size}x{size} matrix:")
    elements = [float(input(f"Element at position ({i // size + 1}, {i % size + 1}): ")) for i in range(size**2)]
    matrix = np.array(elements).reshape(size, size)
    return matrix

def PuissanceItérée(A, tolérance, nombre_max_iterations):
    n = A.shape[0]
    vecteur_initial = np.random.rand(n)
    lambda_prev = 0
    
    for _ in range(1, nombre_max_iterations + 1):
        vecteur_propre_estimé = np.dot(A, vecteur_initial)
        lambda_estimé = np.dot(vecteur_initial, vecteur_propre_estimé)
        vecteur_initial = vecteur_propre_estimé / np.linalg.norm(vecteur_propre_estimé)
        
        if abs(lambda_estimé - lambda_prev) < tolérance:
            return lambda_estimé, vecteur_initial
        
        lambda_prev = lambda_estimé
    
    return lambda_estimé, vecteur_initial

def Déflation(A, lmbda, vecteur_propre, tolérance, nombre_max_iterations):
    n = A.shape[0]
    outer_product = np.outer(vecteur_propre, vecteur_propre)
    B = A - lmbda * outer_product
    
    lambda2, vecteur2 = PuissanceItérée(B, tolérance, nombre_max_iterations)
    return lambda2, vecteur2

def main():
    size = int(input("Enter the size of the square matrix: "))
    tolerance = float(input("Enter the tolerance: "))
    max_iterations = int(input("Enter the maximum number of iterations: "))
    
    # Create the matrix
    matrix = create_matrix(size)
    
    print("Entered matrix:")
    print( "A =",matrix)
    
    # Perform operations
    lambda1, vector1 = PuissanceItérée(matrix, tolerance, max_iterations)
    print(f"\nFirst eigenvalue: {lambda1}")
    print("Corresponding eigenvector:")
    print(vector1)
    
    lambda2, vector2 = Déflation(matrix, lambda1, vector1, tolerance, max_iterations)
    print(f"\nSecond eigenvalue after deflation: {lambda2}")
    print("Corresponding eigenvector after deflation:")
    print(vector2)

if __name__ == "__main__":
    main()