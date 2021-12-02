def isNotSubword(x, idx, sub = '##'):
    return sub not in x[idx] and idx > 0 and idx < len(x) - 1 and sub not in x[idx-1] and sub not in x[idx+1]

def cutting_subword(X, y, size=256):
    res_X, res_y = [], []
    punct = '.!?'
    st = 0
    cur = 0
    
    while (st < len(X)-size):
        flag = True
        for i in range(st+size-1, st-1, -1):
            if X[i] in punct and y[i] == 'O':
                cur = i+1
                flag = False
                break
        if flag:
            for i in range(st+size-1, st-1, -1):
                if isNotSubword(X, i):
                    cur = i+1
                    if y[i] == 'O':
                        cur = i+1
                        break
                
        res_X.append(X[st: cur])
        res_y.append(y[st: cur])
        st = cur

    res_X.append(X[cur:])
    res_y.append(y[cur:])
    return res_X, res_y
