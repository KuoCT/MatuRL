import torch as T


if T.cuda.is_available():
    DEVICE = T.device("cuda")
elif T.backends.mps.is_available():
    DEVICE = T.device("mps")
else:
    DEVICE = T.device("cpu")
