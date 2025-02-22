import random


numbers = []
for i in range(10):
    number = random.randint(1,1000)
    text = number.to_bytes
    numbers.append(text)
    print(text)


for row in numbers:
    print(row)
    
average = [ number.toGH /len(numbers) for number in numbers]
print(len(numbers))

panagiotopoulosenstolos1-8