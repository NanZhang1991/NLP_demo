import sys
 
n = int(input())
arr = sys.stdin.readline().strip().split()
if len(arr) != n:

    pass
print(" ".join(sorted(arr)))