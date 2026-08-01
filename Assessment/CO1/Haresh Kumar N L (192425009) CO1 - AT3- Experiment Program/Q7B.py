w = 0.5
lr = 0.1
target = 1
output = 0.8
gradient = (output-target)
new_weight = w-lr*gradient
print("Updated Weight =",new_weight)
