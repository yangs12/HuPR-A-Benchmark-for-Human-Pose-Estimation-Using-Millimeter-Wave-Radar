import os

num = 276

# os.mkdir('data')
# os.mkdir('data/HuPR')
# os.mkdir('visualization')
# os.mkdir('logs')
# os.mkdir('preprocessing/raw_data')
# os.mkdir('preprocessing/raw_data/iwr1843')
os.makedirs('data', exist_ok = True)
os.makedirs('data/HuPR', exist_ok = True)
os.makedirs('visualization', exist_ok = True)
os.makedirs('logs', exist_ok = True)
os.makedirs('preprocessing/raw_data', exist_ok = True)
os.makedirs('preprocessing/raw_data/iwr1843', exist_ok = True)

for i in range(1, num+1):
    root = 'data/HuPR/'
    dirName = root + 'single_' + str(i)
    dirVertName = dirName + '/vert'
    dirHoriName = dirName + '/hori'
    dirAnnotName = dirName + '/annot'
    dirVisName = dirName + '/visualization'
    os.mkdir(dirName)
    os.mkdir(dirVertName)
    os.mkdir(dirHoriName)
    os.mkdir(dirAnnotName)
    os.mkdir(dirVisName)
