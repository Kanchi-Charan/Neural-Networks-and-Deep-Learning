#Question 01 A:

st = input()  
st = st[:1]+st[2:]  #Removes 2nd character from the string(Slicing the string and attaching again)
st = st[:1]+st[2:]
st = st[::-1]       #Rverse the string
print(st)


