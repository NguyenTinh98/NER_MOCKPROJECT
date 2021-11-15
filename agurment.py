# B_PER = ['anh', 'chị', 'em', 'cô', 'dì', 'gìa', 'bác', 'chú', 'mợ', 'cái', 'thằng']
NAME = ['Mạnh', 'Nghĩa', 'Nam', 'Lợi', 'Tình', 'Hoàng Nguyên', 'Phạm Mạnh']
V = ['đi', 'chơi', 'đang', 'đã', 'ngủ', 'uống', 'học', 'giúp', 'nằm', 'đi', 'đứng', 'yêu', 'ghét', 'kính_trọng']
N = ['cơm', 'du_lịch', 'nhà', 'xe_đạp', 'xe_máy', 'ô_tô', 'taxi', 'con chuột', 'hoa_hồng', 'bàn']

from pyvi import ViTokenizer, ViPosTagger

sequences = []
for name in NAME:
    sent = []
    pos_name = ViPosTagger.postagging(name)[1][0]
    sent.append((name, pos_name, 'B-PER' ))
    for verb in V:
        pos_verb = ViPosTagger.postagging(verb)[1][0]
        sent.append((verb, pos_verb, 'O' ))
        for noun in N:
            pos_noun = ViPosTagger.postagging(noun)[1][0]
            sent.append((noun, pos_noun, 'O' ))
            sequences.append(sent)
            sent.remove((noun, pos_noun, 'O' ))
        sent.remove((verb, pos_verb, 'O' ))
    sent.remove((name, pos_name, 'B-PER' ))