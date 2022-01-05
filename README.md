# 简易新闻文档搜索引擎

该项目使用`Python`语言开发，实现了简单的搜索引擎功能，主要包含布尔检索和简单的语义检索

## 运行环境

开发即测试环境均为`Windows 10`，开发工具选择`PyCharm`，使用了`Anaconda`做包管理，其`Python`版本为`3.8.1`。

## 运行方式

在项目`src`目录下，有三个`.py`文件分别为`Source.py`，`boolean_retrieval.py`，以及`semantic_search.py`。其功能和使用方法大致如下：

* 运行`Source.py`，即可对原始文档进行初始化处理，生成处理后的文档，并进行倒排索引表以及`tf-idf`矩阵的生成和存储
* 运行`boolean_retrieval.py`，即可进行布尔检索，具体演示见实验报告
* 运行`semantic_search.py`，即可进行简单语义检索，具体演示见实验报告

## 关键函数说明

在`Source.py`中，除了`main()`函数之外，总共有19个不同功能的函数，其中关键函数的功能已在实验报告中提及，这里不多赘述。

## 生成文件说明

在`.\output`中，我们生成了这些文件（夹）：

* 文件夹`Edited_dataset`：存储了经过初始化处理之后各篇文档的单词集合
* `courpus.json`：由所给文章生成的语料库，用于进行`tf-idf`值的计算
* `file_name_list.json`：我们建立了一个列表``file_name_list``，表项为一个二元组，分别指示了原文件的相对路径和与其对应的预处理得到的单词列表的相对路径，其顺序也就是我们给予每篇文章的编号
* `inverted_table`：倒排索引表，以字典形式存储
* `tf_idf_matrix.json`：`tf-idf`矩阵，以二级字典形式存储
* `word_dict.json`：一个从单词到其编号的字典
* `word_idf_dict.json`：单词和其`idf`值对应的字典
* `word_list.json`：一个从单词到其编号的字典``word_dict``

