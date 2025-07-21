import os
import Levenshtein


lines = open('onnx_output.txt', 'r').readlines()
lines = [line.strip() for line in lines]
correct = 0
incorrect = 0
total_ned = 0

for line in lines:
    imgname, label, pred, _ = line.split('$$$')
    if label == pred:
        correct += 1
    else:
        incorrect += 1
        
    ned = Levenshtein.distance(label, pred)/max(len(label), len(pred))
    total_ned = ned + total_ned
        
print(f'Correct: {correct}, Incorrect: {incorrect}')
print(f'1-NED: {1 - (total_ned/(correct+incorrect))}')
print(f'Accuracy: {correct/(correct+incorrect)}')
