def convert_spanformat(arr):
    if len(arr) < 1:
        return None
    text = ' '.join([i for i, j in arr])
    pos = 0
    start_end_labels = []
    for word, tag in arr:
        if len(start_end_labels) > 0 and tag == start_end_labels[-1][2]:
            temp = [start_end_labels[-1][0], pos+len(word), tag]
            start_end_labels[-1] = temp.copy()
        else:
            temp = [pos, pos+len(word), tag]
            start_end_labels.append(temp)
        pos += len(word) + 1

    res = dict()   
    for s, e, l in start_end_labels:
        if l != 'O':
            if l not in res:
                res[l] = [(s, e)]
            else:
                res[l].append((s, e))
    return res
 
def compare_span(span1, span2, res, strict= 1):
    all_labels = set(list(span1.keys()) + list(span2.keys()))
    for l in all_labels:
        if l not in res:
            res[l] = [0, 0, 0, 0]
        if l not in span1:
            res[l][3] += len(span2[l])
            continue
        if l not in span2:
            res[l][2] += len(span1[l])
            continue
        res[l][2] += len(span1[l])
        res[l][3] += len(span2[l])
        for s, e in span1[l]:
            for s1, e1 in span2[l]:
                if strict == 2:
                    if s == s1 and e == e1:
                        res[l][0] += 1
                        res[l][1] += 1
                else:
                    temp0, temp1 = iou_single(s, e, s1, e1)
                    if strict == 1:
                        temp0, temp1 = int(temp0), int(temp1)
                    res[l][0] += temp0
                    res[l][1] += temp1
    return res
 
def iou_single(s1, e1, s2, e2):
    smax = max(s1, s2)
    emin = min(e1, e2)
    return max(0, emin - smax) / (e1 - s1) if e1 - s1 > 0 else 0, max(0, emin - smax) / (e2 - s2) if e2 - s2 > 0 else 0
 
# (token - True - pred) 
# [[ ],[ ]]           
def span_f1(arr, strict=1, labels= None, digits=4):
    all_labels = set()
    dictt = dict()
    for ar in arr:
        text, gt, pred = list(zip(*ar))
        gtSpan = convert_spanformat(list(zip(text, gt)))
        predSpan = convert_spanformat(list(zip(text, pred)))
        dictt = compare_span(predSpan, gtSpan, dictt, strict)

        all_labels.update(list(gtSpan.keys()))
    classfication_rp = dict()
    # print(dictt)
    f1_avg = 0
    if labels is None:
        labels = all_labels
    for i in labels:
        precision = dictt[i][0] / dictt[i][2] if dictt[i][2] > 0 else 0
        recall = dictt[i][1] / dictt[i][3] if dictt[i][3] > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        classfication_rp[i] = {'precision': round(precision, digits), 'recall': round(recall, digits), 'f1': round(f1, digits), 'support': dictt[i][3]}
        f1_avg += f1
    return f1_avg / len(labels), classfication_rp
