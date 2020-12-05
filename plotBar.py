import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

normalMalloc = np.array([31.672289, 133.229599, 586.336914, 2628.969971, 13398.737305, 0.0])
unifiedWprefetch = np.array([29.952415, 122.984543, 638.048523, 2542.854004, 12831.206055, 77526.773438])
unified = np.array([31.186975, 128.200409, 554.832764, 2590.423096, 12503.641602, 74978.882812])
binslst = [1024, 2048, 4096, 8192, 16384, 32768]

for i, item in enumerate(normalMalloc):
    if i == 5:
        tmp = unified[i]
        unified[i] = unified[i]/tmp
        unifiedWprefetch[i] = unifiedWprefetch[i]/tmp
    else:
        tmp = normalMalloc[i]
        normalMalloc[i] = normalMalloc[i]/tmp
        unifiedWprefetch[i] = unifiedWprefetch[i]/tmp
        unified[i] = unified[i]/tmp

formatted_arr = []
for i, item in enumerate(binslst):
    formatted_arr.append([binslst[i], normalMalloc[i], unified[i], unifiedWprefetch[i]])
columns = ["X", "normal", "Unified", "Unified/Prefetch"]
df_out = pd.DataFrame(data=formatted_arr, columns=columns)

df_out.plot(x = "X", y = list(set(df_out.columns) - set("X")), kind="bar", stacked=True)
plt.show()
