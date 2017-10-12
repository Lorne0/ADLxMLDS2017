import numpy as np

def edit_distance(a, b):
    m, n = len(a), len(b)
    D = np.zeros((m+1,n+1), dtype=np.int32)
    for i in range(m+1): 
        D[i][0]=i
    for j in range(n+1): 
        D[0][j]=j
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 1 if a[i-1]!=b[j-1] else 0
            D[i][j] = min(D[i-1][j]+1, D[i][j-1]+1, D[i-1][j-1]+cost)

    return D[m][n]

def main():
    print(edit_distance("HelloWorld", "Halloworld"))
    print(edit_distance("AGCCT", "ATCT"))

if __name__=="__main__":
    main()


