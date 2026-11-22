from ragscope.embed import make_embedder

e = make_embedder()

v = e.embed([
    "cat",
    "kitten",
    "dog",
    "puppy",
    "apple",
    "banana",
    "fruit",
    "car",
    "vehicle",
    "computer"
])

print(e.model_name, v.shape)

print(v[0] @ v[1])  # cat vs kitten
print(v[0] @ v[2])  # cat vs dog
print(v[0] @ v[4])  # cat vs apple
print(v[4] @ v[5])  # apple vs banana
print(v[4] @ v[6])  # apple vs fruit
print(v[7] @ v[8])  # car vs vehicle
print(v[0] @ v[9])  # cat vs computer
