# -*- coding: utf-8 -*-
# Alg1 Design a Θ(m3n3) time Brute Force algorithm for solving Problem1
#Task1 Give an implementation of Alg1

#The valid_submatrix1 function iterates over the submatrix defined by r1, c1, r2, and c2 and returns false if it contains an element less than h
def valid_submatrix1(p, r1, c1, r2, c2, h):
    for i in range(r1, r2 + 1):
        for j in range(c1, c2 + 1):
            if p[i][j] < h:
                return False # return false if the submatrix contains an element less than h
    return True # return true if it is a valid submatrix

# return the top left and bottom right indices of the resultant square submatrix
def find_p1_m3n3(p, h):
    m, n = len(p), len(p[0])
    max_size = 0
    top_left, bottom_right = (0, 0), (0, 0)

    #iterate through every possible submatrix and if it is a square submatrix, check if it is a valid result
    for r1 in range(m):
        for c1 in range(n):
            for r2 in range(r1, m):
                for c2 in range(c1, n):
                    if r2 - r1 == c2 - c1:  # Ensure the submatrix is square
                        if valid_submatrix1(p, r1, c1, r2, c2, h):
                            size = (r2 - r1 + 1) * (c2 - c1 + 1)
                            if size > max_size: #get indices of the largest square submatrix
                                max_size = size
                                top_left, bottom_right = (r1+1, c1+1), (r2+1, c2+1)

    return top_left + bottom_right

"""## Task2 Give an implementation of Alg2 (m2n2 algo p1)"""

#Alg2 Design a Θ(m2n2) time algorithm for solving Problem1
#Task2 Give an implementation of Alg2

# calculate elements in each prefix submatrix
def preprocess(p, h):
    m, n = len(p), len(p[0])
    count = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            count[i][j] = count[i - 1][j] + count[i][j - 1] - count[i - 1][j - 1] + (p[i - 1][j - 1] >= h)

    return count

# Iterate over all coordinates representing top left and bottom right indices and return count
def valid_submatrix(count, r1, c1, r2, c2):
    return count[r2 + 1][c2 + 1] - count[r1][c2 + 1] - count[r2 + 1][c1] + count[r1][c1] == (r2 - r1 + 1) * (c2 - c1 + 1)

#Generate submatrices and check if it is a square
def find_p1_m2n2(p, h):
    m, n = len(p), len(p[0])
    max_size = 0
    top_left, bottom_right = (0, 0), (0, 0)
    count = preprocess(p, h)

    for r1 in range(m):
        for c1 in range(n):
            for r2 in range(r1, m):
                for c2 in range(c1, n):
                    if r2 - r1 == c2 - c1:  # Ensure submatrix is square
                        if valid_submatrix(count, r1, c1, r2, c2): # check if submatrix is valid
                            size = (r2 - r1 + 1) * (c2 - c1 + 1) # calculate size of submatrix
                            if size > max_size:       #update max size found
                                max_size = size
                                top_left, bottom_right = (r1+1, c1+1), (r2+1, c2+1)

    return top_left + bottom_right

"""## Task3 Give an implementation of Alg3	(mn dp algo p1)"""

#Alg3 Design a Θ(mn) time Dynamic Programming algorithm for solving Problem1
#Task3 Give an implementation of Alg3
def find_p1__mn_dp(p, h):
    m, n = len(p), len(p[0])
    dp = [[0] * n for _ in range(m)] # creating dp array
    max_size = 0
    top, left, bottom, right = -1, -1, -1, -1
    
    for i in range(m):
        for j in range(n):
            if p[i][j] >= h:
                if i == 0 or j == 0:
                    dp[i][j] = 1 #for the first row and column, set dp[i][j] to 1
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1 # set dp[i][j] to min of dp of its neighbors + 1
                if dp[i][j] > max_size: # update max size
                    max_size = dp[i][j]
                    top, left = i-max_size+2, j-max_size+2
                    bottom, right = i+1, j+1
    
    if max_size == 0:
        return None
    
    return (top, left, bottom, right) # return indices

"""## Task4 Give an implementation of Alg4	(mn2 algo p2)"""

#Alg4 Design a Θ(mn2) time Dynamic Programming algorithm for solving Problem2
#Task4 Give an implementation of Alg4
def find_p2_dp_mn2(p, h):
    rows = len(p)
    cols = len(p[0])
    colDP = [[0] * cols for _ in range(rows)] #initialize cols array to compute longest column of consecutive elements greater than h
    squareDP = [[0] * cols for _ in range(rows)] #initialize square dp to compute size of largest square submatrix 

    # Populating values for squareDP
    for i in range(rows):                            
        for j in range(cols):
            if p[i][j] >= h:
                if i - 1 < 0 or j - 1 < 0:
                    squareDP[i][j] = 1
                else:
                    squareDP[i][j] = min(squareDP[i - 1][j - 1], min(squareDP[i - 1][j], squareDP[i][j - 1])) + 1

  # Populate values for colDP
    for i in range(rows):
        for j in range(cols):
            if p[i][j] >= h:
                if i == 0:
                    colDP[i][j] = 1
                else:
                    colDP[i][j] = colDP[i - 1][j] + 1

    max_side = 2
    max_i = 1
    max_j = 1

  #Calculate the largest submatrix keeping the bottom right corner at the current element
  #First find the length of the longest column of consecutive elements greater than or equal to h that ends at the previous row of that column using colDP
  #   and the length of the longest row of consecutive elements greater than or equal to h that ends at the previous column of that row using bottom_row.
  #Then take the minimum of these two values to get the maximum possible side length of the submatrix
    for i in range(2, rows):
        for j in range(2, cols):
            bottom_row = 0
            bottom_j = j - 1
            while bottom_j >= 0 and p[i][bottom_j] >= h:
                bottom_row += 1
                bottom_j -= 1

            side = min(colDP[i - 1][j], bottom_row)
            sq_length = squareDP[i - 1][j - 1]

# Calculate Side length of the submatrix
            final_side = 0

            if side < sq_length:
                final_side = side + 2
            else:
                top_row = 0
                top_j = j - 1
                while i - 1 - sq_length >= 0 and top_j >= 0 and p[i - 1 - sq_length][top_j] >= h:
                    top_row += 1
                    top_j -= 1

                if i - 1 - sq_length >= 0 and j - 1 - sq_length >= 0 and top_row >= sq_length and colDP[i - 1][j - 1 - sq_length] >= sq_length:
                    final_side = sq_length + 2
                else:
                    final_side = sq_length + 1
# Update max_side if final side is greater 
            if final_side > max_side:
                max_side = final_side
                max_i = i + 1
                max_j = j + 1
# return indices
    return (max_i - max_side + 1, max_j - max_side + 1, max_i, max_j)

"""## Task5a Give a recursive implementation of Alg5 using Memoization	(mn dp p2)"""

#Alg5 Design a Θ(mn) time Dynamic Programming algorithm for solving Problem2
#Task5a Give a recursive implementation of Alg5 using Memoization

def problemTwoMNTimeComplexityWithMemo(p, h):
    m, n = len(p), len(p[0])
    #define dp array
    dp = [[-1] * n for _ in range(m)]
    max_size = 0
    top, left, bottom, right = -1, -1, -1, -1

# recursive function to return max size of submatrix
    def rec(i, j):
        if i == 0 or j == 0:
            return 1
        if dp[i][j] != -1:
            return dp[i][j]
        up, left = rec(i - 1, j), rec(i, j - 1)
        up_left = rec(i - 1, j - 1)
        # check all adjacent neighbors but not corners and update dp array accordingly
        if p[i][j] >= h and p[i+1][j] >= h and p[i-1][j] >= h and p[i][j-1] >= h and p[i][j+1] >= h:
            dp[i][j] = min(up, left, up_left) + 1
        else:
            dp[i][j] = min(up, left, up_left)
        return dp[i][j]

# Find the max size submatrix
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            size = rec(i, j)
            if size > max_size: # update max size
                max_size = size
                top, left = i - max_size + 1, j - max_size + 1
                bottom, right = i + 1, j + 1

    if max_size == 0:
        return None

# return indices
    return (top + 1, left + 1, bottom + 1, right + 1)

"""## Task5b Give an iterative BottomUp implementation of Alg5	(mn dp p2)"""

#Alg5 Design a Θ(mn) time Dynamic Programming algorithm for solving Problem2
#Task5b Give an iterative BottomUp implementation of Alg5
def problemTwoMNTimeComplexity(p, h):
    m, n = len(p), len(p[0])
    dp = [[0] * n for _ in range(m)]
    max_size = 0
    top, left, bottom, right = -1, -1, -1, -1
    
    #Set initial values for the dp array
    for i in range(m):
      for j in range(n):
        if i == 0 or j == 0:
          dp[i][j] = 1 
        elif i == 1 or j == 1: 
          dp[i][j] = 2
        
    # Update dp values by comparing with the adjacent elements but not corners. Update dp[i+1][j+1] based on this.
    for i in range(1,m-1):
        for j in range(1,n-1):
          if p[i][j] >= h and p[i+1][j] >= h and p[i-1][j] >= h and p[i][j-1] >= h and p[i][j+1] >= h:
            dp[i+1][j+1] = min(dp[i][j], dp[i+1][j], dp[i][j+1]) + 1
          else:
            dp[i+1][j+1] = min(dp[i][j], dp[i+1][j], dp[i][j+1])
          # Update max size based on dp array
          if dp[i+1][j+1] > max_size:
            max_size = dp[i+1][j+1]
            top, left = i-max_size+2, j-max_size+2
            bottom, right = i + 1, j +1   
    if max_size == 0:
        return None

# return indices
    return (top + 1, left + 1, bottom + 1, right + 1)

"""## Task6 Give an implementation of Alg6	(m3n3 brute force p3)"""

#Alg6 Design a Θ(m3n3) time Brute Force algorithm for solving Problem3
#Task6 Give an implementation of Alg6

def find_p3_bf(p, h, k):
    m, n = len(p), len(p[0])
    max_area = 0
    indices = None

# find all possible submatrices 
    for i in range(m):
        for j in range(n):
            for x in range(i, m):
                for y in range(j, n):
                    if x - i == y - j:  # Ensure submatrix is square
                        area = (x - i + 1) * (y - j + 1)
                        # Check and update max area
                        if area <= max_area:
                            continue
                        
                        # increment count of elements less than h
                        count = 0
                        for r in range(i, x + 1):
                            for c in range(j, y + 1):
                                if p[r][c] < h:
                                    count += 1
                      #check if the submatrix has only at most k elements less than h
                        if count <= k:
                            max_area = area
                            indices = (i+1, j+1, x+1, y+1)
# return indices
    return indices

"""## Task7a Give a recursive implementation of Alg7 using Memoization	(mnk dp p3)"""

#Alg7 Design a Θ(mnk) time Dynamic Programming algorithm for solving Problem3
#Task7a Give a recursive implementation of Alg7 using Memoization
def find_p3_memo(p, h, k):
    m, n = len(p), len(p[0])
    dp = [[0] * n for _ in range(m)]  # count of elements less than h in each submatrix
    #Intialize dp array 
    for i in range(m):
        for j in range(n):
            dp[i][j] = (dp[i-1][j] if i > 0 else 0) + (dp[i][j-1] if j > 0 else 0) - (dp[i-1][j-1] if i > 0 and j > 0 else 0) + (p[i][j] < h)
    #Initialize memo tuples
    memo = {}
    #Create helper function
    def helper(i, j, k1):
        if (i, j, k1) in memo:
            return memo[(i, j, k1)]
        if k1 == 0:
            res = (i, j, i, j)
        #check that the number of elements less than h is at most only k elements
        elif dp[i+k1][j+k1] - (dp[i+k1][j-1] if j > 0 else 0) - (dp[i-1][j+k1] if i > 0 else 0) + (dp[i-1][j-1] if i > 0 and j > 0 else 0) <= k:
            res = (i, j, i+k1, j+k1)
        else:
            res = helper(i, j, k1-1) # recursive helper call
        memo[(i, j, k1)] = res
        return res
    # create res
    res = None
    # For each possible values of k1, i, j call helper function
    for k1 in range(k+1):
        for i in range(m-k1):
            for j in range(n-k1):
                cur_res = helper(i, j, k1)
                if cur_res is not None and (res is None or func(cur_res) > func(res)):
                    res = cur_res
    res_one_index = None
    res_one_index = tuple(x+1 for x in res)
    return res_one_index  # return bounding indices

def func(a):
    return a[2]-a[0]+1    #returns area bounded by the indices

"""## Task7b Give an iterative BottomUp implementation of Alg7	(mnk dp p3)"""

#Alg7 Design a Θ(mnk) time Dynamic Programming algorithm for solving Problem3
#Task7b Give an iterative BottomUp implementation of Alg7
def find_p3_dp(p, h, k):
    m, n = len(p), len(p[0])
    
    # Intialize empty count array 
    cnt = [[0] * n for _ in range(m)]  # count of elements less than h in each submatrix
    # Populate the array based on the recurrence relation
    for i in range(m):
        for j in range(n):
            cnt[i][j] = (cnt[i-1][j] if i > 0 else 0) + (cnt[i][j-1] if j > 0 else 0) - (cnt[i-1][j-1] if i > 0 and j > 0 else 0) + (p[i][j] < h)
    
    # Create 3d dp array
    dp = [[[None] * (k + 1) for _ in range(n)] for _ in range(m)]
    
    # dp[i][j][k1] represnts the submatrix with maximum size k1 that has at most k elements less than k
    for k1 in range(k + 1):
        for i in range(m):
            for j in range(n):
                if k1 == 0:
                    dp[i][j][k1] = (i, j, i, j)
                elif i + k1 < m and j + k1 < n and cnt[i+k1][j+k1] - (cnt[i+k1][j-1] if j > 0 else 0) - (cnt[i-1][j+k1] if i > 0 else 0) + (cnt[i-1][j-1] if i > 0 and j > 0 else 0) <= k:
                    dp[i][j][k1] = (i, j, i+k1, j+k1)
                else:
                    dp[i][j][k1] = dp[i][j][k1-1]
    
    #Find the maximum size submatrix
    res = None
    for k1 in range(k + 1):
        for i in range(m - k1):
            for j in range(n - k1):
                cur_res = dp[i][j][k1]
                if cur_res is not None and (res is None or func(cur_res) > func(res)):
                    res = cur_res
    res_one_index = None
    res_one_index = tuple(x+1 for x in res)
    return res_one_index # return indices

# compute size of submatrix
def func(a):
    return a[2] - a[0] + 1

"""## Test Cases"""

# p1-1
p11 = [[6,10,9,12], 
	[8,8,8,2], 
	[1,0,0,10],
	[9,10,9,9]]
h11 = 8
#expected output : (1, 2, 2, 3)

#p1-2
p12 = [[2, 4, 2, 5, 5, 5, 5, 5, 3, 2],
[2, 1, 3, 5, 5, 5, 5, 4, 1, 2],
[4, 3, 2, 5, 5, 5, 5, 4, 3, 1],
[2, 4, 2, 5, 5, 5, 5, 1, 3, 2]]
h12 = 5
#expected output : 1 4 4 7

p13 = [[2]]
h13 = 1
#expected output : 0 0 0 0

p14 =[[3, 4, 5, 6, 7],
     [1, 9, 2, 8, 1],
     [5, 10, 7, 6, 4],
     [1, 5, 2, 6, 1],
     [7, 8, 9, 10, 3],
     [2, 1, 7, 5, 13],
     [3, 5, 7, 6, 8],
     [9, 3, 12, 6, 10],
     [2, 1, 7, 6, 4],
     [4, 5, 6, 7, 8]]
h14 = 5 #Expected output : (5,2,7,4) - Not working for m2n2 and m3n3  -> not checking if its a square matrix. 

p15 = [[6, 4, 7, 2, 8, 1],
     [9, 5, 14, 6, 2, 1],
     [7, 13, 8, 11, 5, 9],
     [12, 6, 11, 9, 7, 5],
     [8, 12, 5, 7, 9, 3],
     [3, 1, 9, 5, 6, 4]]
h15 = 5
# Expected output : (1, 0, 4, 3)

p16 = [[12,12,12,3],[22,21,22,2],[22,14,13,10],[14,15,25,3]]
h16 = 10
# Expected output : (0, 0, 2, 2)

p17 = [[3,2,7,1,7,8],
       [1,7,8,9,7,6],
			 [7,9,8,6,8,7],
			 [5,1,2,8,6,7]]
h17 = 5

print(find_p1__mn_dp(p11, h11))
print(find_p1_m2n2(p11, h11))
print(find_p1_m3n3(p11, h11))

print(find_p1__mn_dp(p12, h12))
print(find_p1_m2n2(p12, h12))
print(find_p1_m3n3(p12, h12))

print(find_p1__mn_dp(p13, h13))
print(find_p1_m2n2(p13, h13))
print(find_p1_m3n3(p13, h13))

print(find_p1__mn_dp(p14, h14))
print(find_p1_m2n2(p14, h14))
print(find_p1_m3n3(p14, h14))

print(find_p1__mn_dp(p15, h15))
print(find_p1_m2n2(p15, h15))
print(find_p1_m3n3(p15, h15))

print(find_p1__mn_dp(p16, h16))
print(find_p1_m2n2(p16, h16))
print(find_p1_m3n3(p16, h16))

print(find_p1__mn_dp(p17, h17))
print(find_p1_m2n2(p17, h17))
print(find_p1_m3n3(p17, h17))

#p2-1
p21 = [[4,1,4,6,3,9], 
	[2,4,11,16,5,1],
 	[2,12,41,14,5,3],
	[8,5,13,4,2,3],
	[3,5,4,1,3,5],
	[1,5,3,4,6,1]]
h21 = 10
# expected output : (2, 2, 4, 4)

#p2-2
p22 = [[2,4,2,5,5,5,1,1,3,2], 
	[2,1,3,5,5,5,5,4,1,2],
	[4,3,2,5,5,5,5,5,3,1],
	[2,4,2,5,5,5,5,1,3,2]]
h22 = 5
#expected output (1,4,4,7)

#p2-1
p23 = [[1,1,1,1,1,1], 
	[1,1,1,1,1,1],
 	[1,1,1,1,1,1],
	[1,1,1,1,10,1],
	[1,1,1,10,10,10],
	[1,1,1,1,10,1]]
h23 = 10
# expected output : (3, 3, 5, 5)

#p2-1
p24 = [[1,10,1,1,1,1], 
	[10,10,10,1,1,1],
 	[1,10,1,1,1,1],
	[1,1,1,1,10,1],
	[1,1,1,10,10,10],
	[1,1,1,1,10,1]]
h24 = 10
# expected output : (0, 0, 2, 2)

p25 = [[1,10,10,10,1,1], 
	[10,10,10,10,1,1],
 	[10,10,10,10,1,1],
	[1,10,10,10,1,1],
	[1,1,1,10,10,10],
	[1,1,1,1,10,1]]
h25 = 10

p26 = [[1,1,1,1,1,1], 
	[1,10,10,1,1,1],
 	[10,10,10,10,1,1],
	[1,10,10,1,1,1],
	[1,1,1,1,1,1],
	[1,1,1,1,1,1]]
h26 = 10

p27 = [[1,1,1,1,1,1], 
	[14,10,14,1,1,1],
 	[10,10,10,1,1,1],
	[14,10,14,1,1,1],
	[1,1,1,1,1,1],
	[1,1,1,1,1,1]]
h27 = 10

p28 = [[3,2,7,1,7,8],
       [1,7,8,9,7,6],
			 [7,9,8,6,8,7],
			 [5,1,2,8,6,7]]
h28 = 5

# expected output : (0, 0, 2, 2)

print(find_p2_dp_mn2(p21, h21))
print(problemTwoMNTimeComplexityWithMemo(p21, h21))
print(problemTwoMNTimeComplexity(p21, h21))

print(find_p2_dp_mn2(p22, h22))
print(problemTwoMNTimeComplexityWithMemo(p22, h22))
print(problemTwoMNTimeComplexity(p22, h22))

print(find_p2_dp_mn2(p23, h23))
print(problemTwoMNTimeComplexityWithMemo(p23, h23))
print(problemTwoMNTimeComplexity(p23, h23))

print(find_p2_dp_mn2(p24, h24))
print(problemTwoMNTimeComplexityWithMemo(p24, h24))
print(problemTwoMNTimeComplexity(p24, h24))

print(find_p2_dp_mn2(p25, h25))
print(problemTwoMNTimeComplexityWithMemo(p25, h25))
print(problemTwoMNTimeComplexity(p25, h25))

print(find_p2_dp_mn2(p26, h26))
print(problemTwoMNTimeComplexityWithMemo(p26, h26))
print(problemTwoMNTimeComplexity(p26, h26))

print(find_p2_dp_mn2(p27, h27))
print(problemTwoMNTimeComplexityWithMemo(p27, h27))
print(problemTwoMNTimeComplexity(p27, h27))

print(find_p2_dp_mn2(p28, h28))
print(problemTwoMNTimeComplexityWithMemo(p28, h28))
print(problemTwoMNTimeComplexity(p28, h28))

#p3-1
p31 = [[13,14,13,6,4,1], 
	[14,6,14,1,4,7],
	[11,1,12,5,7,2],
	[4,1,6,7,3,1],
	[4,3,6,4,2,1],
	[1,2,3,4,5,6]]
h31 = 10
k31 = 2
#expected output: (0, 0, 2, 2)

# p3-2
p32 = [[2,4,2,1,2,5,5,1,3,2], 
	[2,1,3,5,3,5,5,4,1,2],
	[4,3,2,5,4,5,5,5,3,1],
	[2,4,2,5,5,1,1,1,3,2]]
h32 = 5
k32 = 7
#expected output: (0, 3, 3, 6)

#p2-1
p33 = [[1,1,1,1,1,1], 
	[1,1,1,1,1,1],
 	[1,1,1,1,1,1],
	[1,1,1,10,1,10],
	[1,1,1,10,1,10],
	[1,1,1,10,10,1]]
h33 = 10
k33 = 3
# expected output : (3, 3, 5, 5)

#p3-3
p34 = [[2, 4, 2, 5, 5, 5, 5, 5, 3, 2],
[2, 4, 2, 5, 5, 5, 5, 5, 3, 2],
[2, 1, 3, 5, 5, 5, 5, 14, 1, 2],
[4, 3, 2, 5, 5, 5, 5, 14, 3, 1],
[2, 4, 2, 5, 5, 5, 5, 11, 3, 2],
[2, 4, 2, 5, 5, 5, 5, 11, 3, 2]]
h34 = 5
k34 = 6
#expected output : (0, 2, 5, 7)

#p2-1
p35 = [[1,1,1,1,1,1], 
	[1,1,1,1,1,1],
 	[1,1,1,1,1,1],
	[1,1,10,10,1,10],
	[1,1,10,10,1,10],
	[1,1,10,10,10,1]]
h35 = 10
k35 = 3
# expected output : (3, 1, 5, 3)

p36 = [[1,1,1,1,1,1], 
	[1,1,1,1,1,1],
 	[1,1,1,1,1,1],
	[1,1,1,11,1,1],
	[1,1,1,1,11,10],
	[1,1,1,1,10,11]]
h36 = 10
k36 = 1
# expected output : (4, 4, 5, 5)



print(find_p3_bf(p31, h31, k31))
print(find_p3_memo(p31, h31, k31))
print(find_p3_dp(p31, h31, k31))

print(find_p3_bf(p32, h32, k32))
print(find_p3_memo(p32, h32, k32))
print(find_p3_dp(p32, h32, k32))

print(find_p3_bf(p33, h33, k33))
print(find_p3_memo(p33, h33, k33))
print(find_p3_dp(p33, h33, k33))

print(find_p3_bf(p34, h34, k34))
print(find_p3_memo(p34, h34, k34))
print(find_p3_dp(p34, h34, k34))

print(find_p3_bf(p35, h35, k35))
print(find_p3_memo(p35, h35, k35))
print(find_p3_dp(p35, h35, k35))

print(find_p3_bf(p36, h36, k36))
print(find_p3_memo(p36, h36, k36))
print(find_p3_dp(p36, h36, k36))