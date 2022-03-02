import cmath
import numpy as np

def solve_cubic(a:float, b:float, c:float, d:float, epsilon:float = 1e-6):
    """
    where the cubic is of the form ax^3 + bx^2 + cx + d = 0
    assumes that the coefficients are real.
    algorithm based on the wikipedia article.
    """
    roots = np.zeros(3, dtype=complex)

    Delta0 = b**2 - 3 * a * c
    Delta1 = 2 * b**3 - 9 * a * b * c + 27 * a**2 * d

    if abs(Delta0) < epsilon:
        if abs(Delta1) < epsilon:
            # print("branch 1 tested!")
            return -b / (3 * a) * np.ones(3)
        else:
            # print("branch 2 tested!")
            C = ((Delta1 + abs(Delta1)) / 2)**(1/3)
            if abs(C) < epsilon:
                C = ((Delta1 - abs(Delta1)) / 2)**(1/3)
            roots[0] = -1/(3 * a) * (b + C + Delta0/C)
    else:
        # print("branch 3 tested!")
        C = np.power((Delta1 + cmath.sqrt(Delta1**2 - 4 * Delta0**3)) / 2, 1/3, dtype=complex)
        roots[0] = -1/(3 * a) * (b + C + Delta0/C)
    
    xi = complex(-1/2, cmath.sqrt(3)/2)
    roots[1] = -1/(3 * a) * (b + xi * C + Delta0/(xi * C))
    roots[2] = -1/(3 * a) * (b + xi**2 * C + Delta0/(xi**2 * C))

    return roots

if __name__ == "__main__":
    # check these answers with this:
    # https://www.wolframalpha.com/widgets/view.jsp?id=578d50248844454e46e24e9ed230843d
    print(f"solve_cubic(2, 3, -11, -6) = {solve_cubic(2, 3, -11, -6)}")
    print(f"solve_cubic(1, 2, 3, 4) = {solve_cubic(1, 2, 3, 4)}")
    print(f"solve_cubic(1, 3, 3, 2) = {solve_cubic(1, 3, 3, 2)}")
    print(f"solve_cubic(1, 3, 3, 1) = {solve_cubic(1, 3, 3, 1)}")
    print(f"solve_cubic(2, 2, 31/9, 1) = {solve_cubic(2, 2, 31/9, 1)}")
