import numpy as np
import pickle as pk

num = {}
with open("all_ans.txt") as fp:
    for s in fp:
        #s = f.strip("\n")
        now = s[0]
        count = 0
        for c in s:
            if c == now:
                count += 1
            else:
                if now not in num:
                    num[now] = []
                num[now].append(count)
                now = c
                count = 1
print(len(num))

avg = {}
for c in num:
    avg[c] = int(np.mean(num[c]))

for c in avg:
    print(c, avg[c])

with open("phone_mean_num.pk", "wb") as fw:
    pk.dump(avg, fw, pk.HIGHEST_PROTOCOL)
