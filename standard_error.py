import math


def se(p1, n1, p2, n2):
    term1 = (p1 * (1-p1))/n1
    term2 = (p2 * (1-p2))/n2
    return math.sqrt(term1 + term2)

if __name__ == "__main__":
    print(se(p1=0.646, n1=415, p2=0.581, n2=308))
