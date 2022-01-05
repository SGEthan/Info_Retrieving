import json
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os
import re
import math
from scipy import sparse
import numpy as np
from scipy import spatial
import time

ORIGINAL_ARTICLE_PATH = '../dataset/US_Financial_News_Articles/'
EDITED_TEXT_PATH = '../output/Edited_dataset/'
OUTPUT_PATH = '../output/'
AND = 0
OR = 1
NOT = 2
priority = {'AND': 1, 'OR': 1, 'NOT': 2, '(': 0}
Op = ['AND', 'OR', 'NOT']
bracket = ['(', ')']


def for_every_original_articles(base):
    for root, ds, fs in os.walk(base):
        for d in ds:
            outpath = '../output/Edited_dataset/' + d
            if not os.path.exists(outpath):
                os.mkdir(outpath)
        for f in fs:
            if re.match(r'.*.json', f):
                fullname = os.path.join(root, f)
                yield (fullname, f)


def original_file_op(in_name, name):
    with open(in_name, 'r', encoding='UTF-8') as f:
        in_dict = json.load(f)

    text = in_dict['text']
    edited_text = original_text_op(text)
    out_dict = {'Edited_text': edited_text}

    dir_list = re.split(r'[/\\]', in_name)
    dir_of_file = dir_list[len(dir_list) - 2]

    outpath = EDITED_TEXT_PATH + dir_of_file

    out_name = os.path.join(outpath, 'edited_' + name)

    with open(out_name, 'w', encoding='UTF-8') as f:
        json.dump(out_dict, f)

    return out_name


def original_text_op(text):
    cutwords1 = word_tokenize(text)  # cut words

    interpunctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
    cutwords2 = [word for word in cutwords1 if word not in interpunctuations]  # delete interpunctuations

    cutwords3 = []

    for word in cutwords2:
        cutwords3.append(PorterStemmer().stem(word))  # get stems

    stops = set(stopwords.words("english"))
    cutwords4 = [word for word in cutwords3 if word not in stops]  # delete stopwords

    return cutwords4


def for_every_edited_articles(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if re.match(r'.*.json', f):
                fullname = os.path.join(root, f)
                yield (fullname, f)


def edited_text_ivtable_op(in_name, name, count, total_word_list, inverted_table):
    # Operation to the edited texts, creating the inverted page table
    with open(in_name, 'r', encoding='UTF-8') as f:
        in_dict = json.load(f)

    for word in in_dict['Edited_text']:
        word_list = inverted_table.setdefault(word, [count[0], [count[1]]])
        if word_list != [count[0], [count[1]]]:
            if word_list[1][len(word_list[1]) - 1] != count[1]:
                inverted_table[word][1].append(count[1])
        else:
            total_word_list.append(word)
            count[0] += 1


def create_inverted_table():
    inverted_table = {}
    word_list = []
    count = [0, 0]  # count[0]: word count; count[1]: article count
    for (full_dir, name) in for_every_edited_articles(EDITED_TEXT_PATH):
        edited_text_ivtable_op(full_dir, name, count, word_list, inverted_table)
        print(count[1])
        count[1] += 1

    with open(os.path.join(OUTPUT_PATH, 'inverted_table.json'), 'w', encoding='UTF-8') as f:
        json.dump(inverted_table, f)  # creating a list that indicates every file and their number

    with open(os.path.join(OUTPUT_PATH, 'word_list.json'), 'w', encoding='UTF-8') as f:
        json.dump(word_list, f)  # creating a list that indicates every file and its number

    print('inverted table created')
    return inverted_table


def create_word_dict():
    # create a dictionary from which we can find the number of a word
    with open(os.path.join(OUTPUT_PATH, 'word_list.json'), 'r', encoding='UTF-8') as f:
        word_list = json.load(f)
    i = 0
    word_dict = {}
    for word in word_list:
        word_dict[word] = i
        print(i)
        i += 1

    with open(os.path.join(OUTPUT_PATH, 'word_dict.json'), 'w', encoding='UTF-8') as f:
        json.dump(word_dict, f)


def display_the_titles(lst):
    with open(os.path.join(OUTPUT_PATH, 'file_name_list.json'), 'r', encoding='UTF-8') as f:
        file_name_list = json.load(f)  # creating a list that indicates every file and its number

    for index in lst:
        with open(file_name_list[index][0], 'r', encoding='UTF-8') as f:
            in_dict = json.load(f)
        print(index, ':', in_dict['title'])


def boolean_transfer(k):
    word_list = word_tokenize(k)

    word_list_2 = []
    for word in word_list:
        if (word not in Op) & (word not in bracket):
            word_list_2.append(PorterStemmer().stem(word))
        else:
            word_list_2.append(word)

    edited_list = transform_op(word_list_2)

    return edited_list


def transform_op(word_list):
    word_stack = []
    op_stack = []
    for word in word_list:
        if (word not in Op) & (word not in bracket):
            word_stack.append(word)
        elif word in Op:
            while len(op_stack) > 0:
                if priority[op_stack[len(op_stack) - 1]] >= priority[word]:
                    word_stack.append(op_stack.pop())
                else:
                    break
            op_stack.append(word)
        elif word == '(':
            op_stack.append(word)
        else:
            oprt = op_stack.pop()
            while oprt != '(':
                word_stack.append(oprt)
                oprt = op_stack.pop()

    while len(op_stack) > 0:
        word_stack.append(op_stack.pop())

    op_stack.clear()
    for word in word_stack:
        if word not in Op:
            op_stack.append(word)
        elif word == 'NOT':
            op_1 = op_stack.pop()
            op_stack.append([op_1, NOT])
        elif word == 'AND':
            op_2 = op_stack.pop()
            op_1 = op_stack.pop()
            op_stack.append([op_1, op_2, AND])
        else:
            op_2 = op_stack.pop()
            op_1 = op_stack.pop()
            op_stack.append([op_1, op_2, OR])

    return op_stack[0]


def boolean_retrieval(search, inverted_table):
    print(search)
    if type(search) == str:  # if it's a word
        if search in inverted_table:
            return inverted_table[search][1]
        else:
            return []

    word_list_out = []
    if len(search) == 2:  # NOT case
        tmp_list = boolean_retrieval(search[0], inverted_table)
        word_list_out = list(range(0, 306242))
        for number in tmp_list:
            word_list_out.remove(number)

    else:
        if (type(search[0]) != list) & (type(search[1]) != list):
            word_list_1 = inverted_table[search[0]][1]
            word_list_2 = inverted_table[search[1]][1]
        else:  # recursive call
            word_list_1 = boolean_retrieval(search[0], inverted_table)
            word_list_2 = boolean_retrieval(search[1], inverted_table)

        if search[2] == AND:  # AND case
            word_list_out = [number for number in word_list_1 if number in word_list_2]

        elif search[2] == OR:  # OR case
            word_list_out = word_list_1
            for word in word_list_2:
                if word not in word_list_out:
                    word_list_out.append(word)
            word_list_out.sort()

    return word_list_out


def boolean_search():
    search_string = input('enter ur search statement here:')
    time0 = time.time()
    search = boolean_transfer(search_string)
    with open(os.path.join(OUTPUT_PATH, 'inverted_table.json'), 'r', encoding='UTF-8') as f:
        inverted_table = json.load(f)
    search_result = boolean_retrieval(search, inverted_table)

    with open(os.path.join(OUTPUT_PATH, 'search_result.json'), 'w', encoding='UTF-8') as f:
        json.dump(search_result, f)

    print('Matched file numbers:')
    print(search_result)
    print('And their titles:')
    display_the_titles(search_result)
    print('time used:')
    print(time.time() - time0)


def creating_corpus():
    collection = []
    i = 0
    with open(os.path.join(OUTPUT_PATH, 'file_name_list.json'), 'r', encoding='UTF-8') as f:
        file_name_list = json.load(f)
        for file in file_name_list:
            with open(file[1], 'r', encoding='UTF-8') as g:
                article = json.load(g)
                word_collection = article['Edited_text']
                collection.append(word_collection)
                print(i)
                i += 1

    with open(os.path.join(OUTPUT_PATH, 'corpus.json'), 'w', encoding='UTF-8') as f:
        json.dump(collection, f)
    return collection


def tf(word, word_list):
    return float(word_list.count(word)/len(word_list))


def creating_idf_dict(corpus, word_list):
    idf_dict = {}
    length = len(corpus)

    with open(os.path.join(OUTPUT_PATH, 'inverted_table.json'), 'r', encoding='UTF-8') as f:
        inverted_table = json.load(f)

    word_count = 0
    for word in word_list:
        count = len(inverted_table[word])
        idf = math.log(length/(1+count))
        idf_dict[word] = idf
        print(word_count, idf)
        word_count += 1

    with open(os.path.join(OUTPUT_PATH, 'word_idf_dict.json'), 'w', encoding='UTF-8') as g:
        json.dump(idf_dict, f)  # creating a dictionary that indicated every word and its idf


def tf_idf():
    creating_corpus()
    with open(os.path.join(OUTPUT_PATH, 'corpus.json'), 'r', encoding='UTF-8') as f:
        corpus = json.load(f)

    with open(os.path.join(OUTPUT_PATH, 'word_dict.json'), 'r', encoding='UTF-8') as f:
        word_dict = json.load(f)

    with open(os.path.join(OUTPUT_PATH, 'word_list.json'), 'r', encoding='UTF-8') as f:
        word_list = json.load(f)

    creating_idf_dict(corpus, word_list)
    with open(os.path.join(OUTPUT_PATH, 'word_idf_dict.json'), 'r', encoding='UTF-8') as f:
        idf_dict = json.load(f)

    tf_idf_matrix = {}

    i = 0

    for article in corpus:
        tf_idf_matrix[i] = {}
        for word in article:  # for every word
            if word_dict[word] not in tf_idf_matrix[i]:  # if not computed before
                ti = tf(word, article) * idf_dict[word]
                if ti != 0:
                    tf_idf_matrix[i][word_dict[word]] = ti
                    print((i, word_dict[word], ti))
        i += 1

    with open(os.path.join(OUTPUT_PATH, 'tf_idf_matrix.json'), 'w', encoding='UTF-8') as f:
        json.dump(tf_idf_matrix, f)


def for_every_tf_idf_vec(tf_idf_matrix, word_dict):
    # with open(os.path.join(OUTPUT_PATH, 'tf_idf_matrix.json'), 'r', encoding='UTF-8') as f:
    #     tf_idf_matrix = json.load(f)

    row = []
    column = []
    mtx_value = []
    # with open(os.path.join(OUTPUT_PATH, 'word_dict.json'), 'r', encoding='UTF-8') as f:
    #     word_dict = json.load(f)
    length = len(word_dict)

    for i in range(0, 306241):
        for key in tf_idf_matrix[str(i)]:
            row.append(0)
            column.append(int(key))
            value = tf_idf_matrix[str(i)][key]
            mtx_value.append(value)
        a = sparse.coo_matrix((mtx_value, (row, column)), shape=(1, length))
        row.clear()
        column.clear()
        mtx_value.clear()
        yield a


def insert(lst, pair, max_length):
    length = len(lst)
    if length == 0:
        lst.append(pair)
        return
    elif length == max_length:
        if pair[1] >= lst[max_length-1][1]:
            return

    i = 0
    while lst[i][1] <= pair[1]:
        i += 1
        if i == length:
            break

    if (i == length) & (i < max_length):
        lst.append(pair)
    else:
        lst.insert(i, pair)
        if len(lst) > max_length:
            del lst[max_length]


def semantic_retrieval():
    with open(os.path.join(OUTPUT_PATH, 'word_dict.json'), 'r', encoding='UTF-8') as f:
        word_dict = json.load(f)

    with open(os.path.join(OUTPUT_PATH, 'word_idf_dict.json'), 'r', encoding='UTF-8') as f:
        idf_dict = json.load(f)

    input_string = input('enter ur searching statement here:')
    input_list = word_tokenize(input_string)

    i = []
    j = []
    ti_list = []

    for word in input_list:
        if word in word_dict:
            ti = tf(word, input_list) * idf_dict[word]
            i.append(word_dict[word])
            j.append(0)
            ti_list.append(ti)
            print((0, word, word_dict[word], ti))

    length = len(word_dict)
    a = sparse.coo_matrix((ti_list, (j, i)), shape=(1, length)).toarray()

    i = 0
    least_distance = []
    with open(os.path.join(OUTPUT_PATH, 'tf_idf_matrix.json'), 'r', encoding='UTF-8') as f:
        tf_idf_matrix = json.load(f)

    time0 = time.time()
    for cur_vec in for_every_tf_idf_vec(tf_idf_matrix, word_dict):
        i += 1
        distance = np.linalg.norm(cur_vec.toarray()-a)  # 7.578181028366089s
        # distance = spatial.distance.euclidean(a, cur_vec.toarray())  # 8.908097743988037s
        insert(least_distance, (i, distance), 10)
        print((i, distance))
        if i > 1000:
            print(time.time()-time0)
            break

    title_list = []
    for pair in least_distance:
        print(pair)
        title_list.append(pair[0])

    display_the_titles(title_list)


def main():
    file_name_list = []
    for (full_dir, name) in for_every_original_articles(ORIGINAL_ARTICLE_PATH):
        print(full_dir)
        full_out_name = original_file_op(full_dir, name)
        file_name_list.append((full_dir, full_out_name))

    with open(os.path.join(OUTPUT_PATH, 'file_name_list.json'), 'w', encoding='UTF-8') as f:
        json.dump(file_name_list, f)  # creating a list that indicates every file and its number

    create_inverted_table()

    create_word_dict()

    tf_idf()

    os.system('pause')


if __name__ == '__main__':
    main()
