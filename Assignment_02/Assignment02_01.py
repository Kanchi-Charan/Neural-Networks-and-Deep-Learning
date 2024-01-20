#Question 01

#Function for FULL_name
#It takes two arguments named first and last. 
#It returns the full name by concatinationg both strings.
def fullname(first,last):       
    return first+" "+last

#String_alternative function.
def string_alternative(Full_name):
    ans = ""
    for i in Full_name[::2]:     
        ans+=i
        
    return ans

#Input for first name
First_name = input()

#Input for last name
Last_name = input()

Full_name = fullname(First_name,Last_name) #Function call which returns fullname
print(Full_name)

AlternameString = string_alternative(Full_name)
print(AlternameString)


