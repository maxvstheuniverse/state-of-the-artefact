import numpy as np
import os


seed_path = os.path.join(os.getcwd(), "data", "seeds")

tt_one = np.load(os.path.join(seed_path, "tt_temp_1_20k.npy"))
tt_two = np.load(os.path.join(seed_path, "tt_temp_1_20k+.npy"))

print(tt_one.shape, tt_two.shape)
data = np.append(tt_one, tt_two, axis=0)

unique = np.unique(data, axis=0)
print(len(data), len(unique))
np.save(os.path.join(seed_path, "musicvae_tt_20k.npy"), unique[:20000])