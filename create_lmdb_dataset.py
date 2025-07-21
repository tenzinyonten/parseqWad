import fire
import os
import lmdb
import cv2
import pickle
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def process_image_line(i, line, checkValid):
    try:
        imagePath, label = line.strip('\n').split('$$$')
    except Exception as e:
        print(e)
        return None, None, None

    if not os.path.exists(imagePath):
        print(f'{imagePath} does not exist')
        return None, None, None

    with open(imagePath, 'rb') as f:
        imageBin = f.read()

    if checkValid:
        try:
            if not checkImageIsValid(imageBin):
                print(f'{imagePath} is not a valid image')
                return None, None, None
        except Exception as e:
            print(f'Error occurred at line {i}: {e}')
            with open(outputPath + '/error_image_log.txt', 'a') as log:
                log.write(f'{i}-th image data error: {e}\n')
            return None, None, None

    return imageBin, label, os.path.abspath(imagePath)


def createDataset(gtFile, outputPath, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        gtFile     :  path to ground truth text file. The GT file should contain the absolute path of images.
        outputPath :  path to save LMDB
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    data_split = gtFile.split('/')[-1].split('.')[0]
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1

    with open(gtFile, 'r', encoding='utf-8') as data:
        datalist = data.readlines()

    nSamples = len(datalist)

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_image_line, i, datalist[i], checkValid): i for i in range(nSamples)}

        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue

            imageBin, label, imagePath = result
            if imageBin is None:
                continue

            imageKey = f'image-{cnt:09d}'.encode()
            labelKey = f'label-{cnt:09d}'.encode()
            pathKey = f'path-{cnt:09d}'.encode()
            cache[imageKey] = imageBin
            cache[labelKey] = label.encode()
            cache[pathKey] = imagePath.encode()

            if cnt % 1000 == 0:
                writeCache(env, cache)
                cache = {}
                print(f'Written {cnt} / {nSamples}')
            cnt += 1

    nSamples = cnt - 1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)

    print(f'Created dataset with {nSamples} samples')


if __name__ == '__main__':
    fire.Fire(createDataset)
