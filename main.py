import os
import csv

input_dir = os.environ['SM_INPUT_DIR']
input_text = ''

print('###### opening train')
with open(input_dir + "/data/training/train.csv", 'r') as fp:
    # lines = len(fp.readlines())
    # print('Total Number of lines:', lines)
    input_text = str(len(fp.readlines()))

print('###### closed train')


print('###### opening model')
model_dir = os.environ['SM_MODEL_DIR']
with open(model_dir + '/model.txt', 'w') as f:
    f.write('model cio')
print('###### closing model')
    
output_dir = os.environ['SM_OUTPUT_DIR']
with open(output_dir + '/output.txt', 'w') as f:
    f.write('Ciao output')
with open(output_dir + '/output_csv.txt', 'w') as f:
    f.write(input_text)
    
