def compare_span(span1, span2, res, strict= True):
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
                temp0, temp1 = iou_single(s, e, s1, e1)
                if strict:
                    temp0, temp1 = int(temp0), int(temp1)
                res[l][0] += temp0
                res[l][1] += temp1
    return res
