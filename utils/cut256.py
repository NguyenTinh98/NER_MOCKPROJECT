def isSubword(x, idx, sub = '##'):
    return sub not in x[idx] and idx > 0 and idx < len(x) - 1 and sub not in x[idx-1] and sub not in x[idx+1]

def subword(X, y, size=256):
    res_X, res_y = [], []
    punct = '.!?'
    st = 0
    cur = st + size
    
    while (cur < len(X)):
        flag = True
        for i in range(cur-1, cur-1-size, -1):
            if X[i] in punct and y[i] == 'O':
                res_X.append(X[st: i+1])
                res_y.append(y[st: i+1])
                st = i+1
                cur = st + size
                flag = False
                break
        if flag:
            for i in range(cur-1, cur-1-size, -1):
                if y[i] == 'O' and (i > 0 and y[i-1] == 'O') and (i < len(X) - 1 and y[i+1] == 'O') and isSubword(x, i):
                    res_X.append(X[st: i+1])
                    res_y.append(y[st: i+1])
                    st = i+1
                    cur = st + size
                    break
    res_X.append(X[st:])
    res_y.append(y[st:])
    return res_X, res_y
