# # import numpy as np

# # data = np.load("stereo_params.npz")
# # print(list(data.keys()))



# import numpy as np

# d = np.load("stereo_params.npz")
# print("P1 shape:", d["P1"].shape)
# print("P2 shape:", d["P2"].shape)
# print("K1 shape:", d["K1"].shape)
# print("K2 shape:", d["K2"].shape)


import numpy as np
d = np.load("stereo_params.npz")
R = d["R"]

left_up = np.array([0, -1, 0])  # left cam upward direction
right_up = R @ left_up           # right cam upward in left-cam frame

print("Left cam up =", left_up)
print("Right cam up =", right_up)

avg_up = left_up + right_up
print("Average visual up =", avg_up)
