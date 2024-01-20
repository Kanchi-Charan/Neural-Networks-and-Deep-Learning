Input_file = open("input.txt", "r")
#Write the number of words count and split the data from input file
output_file = open("output.txt", "w")

content = {}


for line in Input_file:
    output_file.write(line)
    new_l = line.split()
    for x in new_l:
        if(content.get(x)==None):
            content[x]=1
        else:
            content[x] = content[x] + 1
            
output_file.write(" Word_Count : ")
for key, value in content.items(): 
        output_file.write('%s:%s\n' % (key, value))
Input_file.close()
output_file.close()