from IPython.display import HTML as html_print

def cstr(s,color='black'):
    '''
        input: 'Tân'
        output: 'Tag HTML of word'
    '''
    return "<text style='color:{};font-size:150%'><b> {} </b></text>".format(color,s)

def print_color(lst):
    '''
        input: list of word, ex:['Chào', 'Tân', 'xinh', 'đẹp']
        output: show color of text
    '''
    return html_print(' '.join(lst))

def visualize(predict_, labels):
    '''
        description: visual dự đoán dạng [(word, label), ....]
    
        :predict_ là một sentence, type là list dạng: [(word, label), ....]
        :labels là một list tất cả các nhãn có thể có trong tập data, dùng để cấu hình màu nhận dạng

        :return "lst" dạng (word, label) kết hợp màu tương ứng với nhãn (gợi ý: print_color(lst) để in ra màn hình
        :return "detail_labels" dạng (detail_labels, color) mô tả màu tương ứng với nhãn (gợi ý: print_color(detail_labels) để in ra màn hình

    '''
    colors = ["Blue","Crimson","Red","Maroon","Chartreuse","Misty Rose","Salmon","Navy","Orange","Teal","Coral","Purple","Gold","Ivory","Yellow","Olive","Yellow","Green","Lawn" "green","Chartreuse"]
    
    # ghép labels với colors
    label_colors = {}
    
    i = 0
    for label in labels:
        label_colors[label] = colors[i]
        i += 1

    detail_labels = []
    for key in label_colors:
        detail_labels.append(cstr(key,label_colors[key]))
    
    # ghép label word dự đoán với colors
    lst =[]
    for predict in predict_:
        word,tag = predict
        if tag == "O":
            lst.append(word)
    
        else:
            lst.append(cstr(word, color= label_colors[tag]))

    return lst, detail_labels