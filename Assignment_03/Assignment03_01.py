#700757114

class Employee:
    employee_count = 0

    def __init__(self,name,family,salary,deparemnt):
        self.name = name
        self.family = family
        self.salary = salary
        self.department = deparemnt
        Employee.employee_count += 1;        #Count of employee instances is increased

    @staticmethod
    def Average_Salary(Employees):
        Total_Salary = sum(employee.salary for employee in Employees)
        return Total_Salary/len(Employees)
    

class FulltimeEmployee(Employee):
    pass


E1 = Employee("Charan","Kanchi",1000,"Science")
E2 = Employee("Anvesh","Amara",3500,"Data science")

fulltime_E1 = FulltimeEmployee("Shireesh","Bandi",2000,"Doctor")
fulltime_E2 = FulltimeEmployee("Shabbu","Guntur",3400,"Artist")

employees = [E1,E2,fulltime_E1,fulltime_E2]
Avg_Sal = Employee.Average_Salary(employees)

print("Count of number of Employees:",Employee.employee_count)
print("Average Salary of a Employee: ",Avg_Sal)

