#700757114


import numpy as np

var = np.random.uniform(1,20,20)            #Declaration of numpy array
reshaped_var = var.reshape(4,5)             #Reshaping the array

reshaped_var[np.arange(4), np.argmax(reshaped_var, axis=1)] = 0

print("Original random vector:")
print(var)
print("\nReshaped array:")
print(reshaped_var)