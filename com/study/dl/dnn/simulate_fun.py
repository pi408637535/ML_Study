
import torch as t
import numpy as np
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
from sklearn.model_selection import train_test_split

def fizz_buzz_encode(i):
    if i % 15 == 0:
        return 3
    elif i % 5 == 0:
        return 2
    elif i % 3 == 0:
        return 1
    else:
        return 0


def fizz_buzz_decode(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]

def fizz_buzz_decode1(i, prediction):
    return [i, 0, 1, 2][prediction]


def helper(i):
    print(fizz_buzz_decode1(i, fizz_buzz_encode(i)))


def binary_encode(i, num_digist):
    return np.array([i >> d & 1 for d in range(num_digist)][::-1])


device = None
def check_gpu(args, logger):
    use_gpu = False
    if use_gpu and t.cuda.is_available():
        cudnn.benchmark = True
        #t.manual_seed(args.seed)
        t.cuda.manual_seed_all(args.seed)
        t.backends.cudnn.deterministic = True
        use_gpu = True
    else:
        logger.info('GPU is highly recommend!')

    device = t.device('cuda' if use_gpu else 'cpu')



if __name__ == '__main__':
    NUM_DIGITS = 10
    trX = t.Tensor([binary_encode(i, NUM_DIGITS) for i in range(100, 2 ** NUM_DIGITS)])
    trY = t.LongTensor([fizz_buzz_encode(i) for i in range(100, 2 ** NUM_DIGITS)])


    NUM_HIDDEN = 400
    model = nn.Sequential(
        nn.Linear(NUM_DIGITS, NUM_HIDDEN),
        nn.ReLU(),
        nn.Linear(NUM_HIDDEN, 4)
    )
    model.to(device)

    loss_fun = nn.CrossEntropyLoss()
    optimizer = t.optim.SGD(model.parameters(), lr = 1e-2)

    BATCH = 128
    for epoch in range(10000):
        for start in range(0, len(trX), BATCH):
            end = start + BATCH
            batchX = trX[start:end]
            batchY = trY[start:end]

            if t.cuda.is_available():
                batchX = batchX.to(device)
                batchY = batchY.to(device)

            y_pred = model(batchX)
            loss = loss_fun(y_pred, batchY)
            print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    trX = t.Tensor([binary_encode(i, NUM_DIGITS) for i in range(0, 100)])
    y_pred = model(trX)
    y = y_pred.max(dim=1)[1].cpu().numpy().tolist()
    x = [i for i in range(0, 100)]
    data = zip(x, y)
    for i, j in data:
        print(fizz_buzz_decode(i, j))