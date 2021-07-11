import random

cnt0 = 0
cnt1 = 0

for i in range(5000):
    a = random.randint(0, 1)
    if a == 0:
        cnt0 += 1
    else:
        cnt1 += 1

print(cnt0, cnt1)




