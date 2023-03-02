# -*- coding: utf-8 -*-
import heapq

"""## STRATEGY 1

Iterate over each day starting from day 1 . . . n. For each day, among the unpainted houses that are available to be painted on that day, paint the house that started being available the earliest.
"""

import heapq
#initially we set the number of days when the painter can paint and create the list of houses that have to be painted.
#Input the days and houses from the user: 
days, m = map(int, input('Enter required number of days (n) and houses to be painted (m): ').split())

#Creating an empty list of houses
houseList = []
for i in range(m):
    start, end = map(int, input().split())
    h = [start, end]
    houseList.append(h)

#we initialize an empty list result to store the final indices and a priority queue to maintain houses available for painting.
result = ''
availableHousepq = [] #priority queue that maintains the available houses.
index = 0 
#Iterating over each day (The time complexity of this will be maximum n times)
for i in range(1, days + 1): 
  # We check if the priority queue contains houses that are not available to be painted on day i and pop these houses.
  # the priority queue operations will run for a max of m times across all iterations and each operation has a time complexity of O(logm)
  if availableHousepq: 
    while availableHousepq and availableHousepq[0][1][0][0] < i and availableHousepq[0][1][0][1] < i: 
      heapq.heappop(availableHousepq)
  # Add the available houses from the list to the priority queue sorted by the earliest start date.
  while index < len(houseList):
    if houseList[index][0] <= i and houseList[index][1] >= i:
      heapq.heappush(availableHousepq, (houseList[index][0], [houseList[index], index]))
      index += 1 #increment the index to the next house in the list
    else:
      break 
  #If a house is available to be painted on that day, pop the priority queue and append the house index to the resultant array.
  if availableHousepq:
    curr = heapq.heappop(availableHousepq)
    houseAtindex = curr[1][1] + 1
    result = result + str(houseAtindex) + ' '

print(result)

"""## STRATEGY 2

Iterate over each day starting from day 1 . . . n. For each day, among the unpainted houses that are available that day, paint the house that started being available the latest.
"""

import heapq
#initially we set the number of days when the painter can paint and create the list of houses that have to be painted.
#Input the days and houses from the user: 
days, m = map(int, input('Enter required number of days (n) and houses to be painted (m): ').split())

#Creating an empty list of houses
houseList = []
for i in range(m):
    start, end = map(int, input().split())
    h = [start, end]
    houseList.append(h)

#we initialize an empty list result to store the final indices and a priority queue to maintain houses available for painting.
result = ''
availableHousepq = [] #priority queue that maintains the available houses.
index = 0 
#Iterating over each day.
for i in range(1, days + 1): 
  # We check if the priority queue contains houses that are not available to be painted on day i and pop these houses.
  # the priority queue operations will run for a max of m times across all iterations and each operation has a time complexity of O(logm)
  if availableHousepq:
    while availableHousepq and availableHousepq[0][1][0][0] < i and availableHousepq[0][1][0][1] < i: 
      heapq.heappop(availableHousepq)
  # Add the available houses from the list to the priority queue sorted by the start date that was available latest.
  while index < len(houseList):
    if houseList[index][0] <= i and houseList[index][1] >= i:
      heapq.heappush(availableHousepq, (-houseList[index][0], [houseList[index], index]))
      index += 1 
    else:
      break 
  #If a house is available to be painted on that day, pop the priority queue and append the house index to the resultant array.
  if availableHousepq:
    curr = heapq.heappop(availableHousepq)
    houseAtindex = curr[1][1] + 1
    result = result + str(houseAtindex) + ' '

print(result)

"""## STRATEGY 3

Iterate over each day starting from day 1 . . . n. For each day, among the unpainted houses that are available that day, paint the house that is available for the shortest duration.
"""

import heapq
#initially we set the number of days when the painter can paint and create the list of houses that have to be painted.
#Input the days and houses from the user: 
days, m = map(int, input('Enter required number of days (n) and houses to be painted (m): ').split())

#Creating an empty list of houses
houseList = []
for i in range(m):
    start, end = map(int, input().split())
    h = [start, end]
    houseList.append(h)

#we initialize an empty list result to store the final indices and a priority queue to maintain houses available for painting.
result = []
availableHousepq = [] #priority queue that maintains the available houses.
index = 0 
#Iterating over each day.
for i in range(1, days + 1): 
  # We check if the priority queue contains houses that are not available to be painted on day i and pop these houses.
  # the priority queue operations will run for a max of m times across all iterations and each operation has a time complexity of O(logm)
  if availableHousepq:
    while availableHousepq and availableHousepq[0][1][0][0] < i and availableHousepq[0][1][0][1] < i: 
      heapq.heappop(availableHousepq)
  # Add the available houses from the list to the priority queue sorted by the duration between the end and start date.
  while index < len(houseList):
    if houseList[index][0] <= i and houseList[index][1] >= i:
      diff = houseList[index][1] - houseList[index][0]
      heapq.heappush(availableHousepq, (diff, [houseList[index], index]))
      index += 1 
    else:
      break 
  #If a house is available to be painted on that day, pop the priority queue and append the house index to the resultant array.
  if availableHousepq:
    curr = heapq.heappop(availableHousepq)
    houseAtindex = curr[1][1] + 1
    result = result + str(houseAtindex) + ' '
print(result)

"""## STRATEGY 4

Iterate over each day starting from day 1 . . . n. For each day, among the unpainted houses that are available that day, paint the house that will stop being available the earliest.
"""

import heapq
#initially we set the number of days when the painter can paint and create the list of houses that have to be painted.
#Input the days and houses from the user: 
days, m = map(int, input('Enter required number of days (n) and houses to be painted (m): ').split())

#Creating an empty list of houses
houseList = []
for i in range(m):
    start, end = map(int, input().split())
    h = [start, end]
    houseList.append(h)

#we initialize an empty list result to store the final indices and a priority queue to maintain houses available for painting.
result = []
availableHousepq = [] #priority queue that maintains the available houses.
index = 0 
#Iterating over each day.
for i in range(1, days + 1): 
  # We check if the priority queue contains houses that are not available to be painted on day i and pop these houses.
  # the priority queue operations will run for a max of m times across all iterations and each operation has a time complexity of O(logm)
  if availableHousepq:
    while availableHousepq and availableHousepq[0][1][0][0] < i and availableHousepq[0][1][0][1] < i: 
      heapq.heappop(availableHousepq)
  # Add the available houses from the list to the priority queue sorted by the ending date in ascending order.
  while index < len(houseList):
    if houseList[index][0] <= i and houseList[index][1] >= i:
      heapq.heappush(availableHousepq, (houseList[index][1], [houseList[index], index]))
      index += 1 
    else:
      break 
  #If a house is available to be painted on that day, pop the priority queue and append the house index to the resultant array.
  if availableHousepq:
    curr = heapq.heappop(availableHousepq)
    houseAtindex = curr[1][1] + 1
    result = result + str(houseAtindex) + ' '

print(result)

"""## BONUS

Design and analyze (time, space and correctness) of an Θ(m log(m)) algorithm based on the greedy strategy that you proved to be Optimal in Section 2.
This strategy is strategy 4 where we pick houses that stop being available the earliest. We use the same condition in this bonus algorithm
"""

# Input the number of days (n) and number of houses (m) from the user
n, m = map(int, input('Enter number of days (n) and number of houses (m) in respective order: ').split())

# Create an empty list of houses and populate it with start and end dates
houseList = []
for i in range(m):
    start, end = map(int, input().split())
    h = [start, end, i + 1]
    houseList.append(h)

# Print the initial list of unpainted houses for verification
print()

sorted_houses = sorted(houseList, key=lambda x: x[1])

result = '' # create an empty string for storing the results
currDay = 1 # initialize the current day to the first day
for i in range(0, len(sorted_houses)): # iterate through the sorted list of houses
  # if the current house is out of range or the current day is greater than the search range, break the loop
  if sorted_houses[i][0] > n or currDay > n: 
    break
  # if the current house starts after the current day, update the current day
  if sorted_houses[i][0] > currDay: 
    currDay = sorted_houses[i][0]
  # if the current house is within range, add its index to the results and update the current day
  if sorted_houses[i][0] <= currDay and sorted_houses[i][1] >= currDay:
    ele = sorted_houses[i]
    currentHouse = ele[2]
    result = result + str(currentHouse) + ' '
    currDay += 1

print(result)