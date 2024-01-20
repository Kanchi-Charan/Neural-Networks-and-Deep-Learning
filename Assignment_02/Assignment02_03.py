#Input the size of the list
n = int(input())
#Empty list
l = []
#For loop which gives input into the list. List of hiehgts in inches.
for i in range(n):
    x = int(input())
    l.append(x)

#Empty output list
li = []
#For loop ehich takes the output into the list. List of heights in centimeters.
for i in range(n):
    li.append(l[i]*2.54)
    print(li[i])
