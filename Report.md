# Web Info Lab 1 Report

## 系统概述

本次实验要求实现``布尔检索``和``语义检索``。

本实验小组成员及学号如下：

* PB19000050-邓一川
* PB19000061-赖铮岩

## 实验过程

### 预处理部分

首先，我们需要对提供的数据集进行预处理，这里选择使用``nltk``包进行分词，词根化和去停用词，分别调用了 ``word_tokenize`` 函数，``PorterStemmer().stem()`` 函数，以及``nltk``提供的英语停用词集。由此，我们对每一篇文章处理得到了一个单词列表，将其以``json``格式分别存入到``.\output\Edited_dataset``中，该目录下的所有文件名称样式为``edited_*.json``，``*``表示原文件名称。

在此过程中，我们同时建立了一个列表``file_name_list``，表项为一个二元组，分别指示了原文件的相对路径和与其对应的预处理得到的单词列表的相对路径，其顺序也就是我们给予每篇文章的编号，存储于硬盘中，其路径为``.\output\Edited_dataset\file_name_list.json``。

### 倒排索引表的建立

在得到预处理的文档集合 $D=\{D_1,D_2,…,D_N\}$，即上述每一篇文章生成的单词列表后，我们开始生成倒排索引表。过程大致如下：

1. 新建立一个字典 ``inverted_table``，其中以每个单词作为字典的键，以一个二元列表作为值，该二元列表格式为``[word_num,[article_num_list:]]``，其中``word_num``存储每个单词的编号，按照遍历的顺序来获得；``[article_num_list:]``为一个列表，存储与单词相关的所有文章编号的列表。
2. 遍历每一篇文章，按照其单词列表，建立倒排索引表，根据单词查找表项并将当前的文章编号加入。
3. 当遍历完成之后，将``inverted_table``以``json``文件存储，其路径为``.\output\Edited_dataset\inverted_table.json``

在倒排索引表建立之后，我们同时发现，如果每次从单词查找其编号时，都访问一次倒排索引表，那么时间开销将非常巨大，而且在当下，从单词编号查找单词的工作无法进行，所以我们单独写了一个函数``created_word_dict()``，它遍历倒排索引表，单独建立一个从单词到其编号的字典``word_dict``，同时建立一个单词列表``word_list``，索引即为其编号，为了此后的访问需要，我们将这两个数据结构也保存于硬盘，路径分别为``.\output\Edited_dataset\word_dict.json``和``.\output\Edited_dataset\word_list.json``

### 布尔检索

对于布尔检索，我们的处理思路和数据访问及处理大致如下：

1. 首先读入一个布尔检索的字符串 $Q_{bool}$，其表达式大致如 $(NOT)\;exp\;(AND|OR\;(NOT)\;exp)$，其中 $exp$ 为表达式，查询表达式之间由空格分隔，或以括号表示整体。

2. 读入之后需要对 $Q_{bool}$ 进行规则化处理，进入一个函数 ``boolean_transfer(k)``，其中``k``为输入的字符串。

   * 首先，对于输入的字符串进行分词处理，和预处理部分相同，调用了 ``word_tokenize`` 函数；

   * 然后我们会将其中除了逻辑符 $AND$， $OR$， $NOT$，以及括号之外的输入符，全部进行词根化，同样的，我们调用``PorterStemmer().stem()`` 函数进行处理；

   * 最后，我们对输出的字符集合进行后缀化，即将输入字符串转为后缀表达式，最后从后缀表达式，我们得到了形如以下格式的查询列表：

     格式：嵌套列表，分为以下三类：

     * 形如 ``[exp,NOT]``：表示 $NOT\;exp$
     * 形如``[exp1,exp2,AND]``：表示 $exp_1\;AND\;exp2$
     * 形如``[exp1,exp2,OR]``：表示 $exp_1\;OR\;exp2$

     其中 ``exp`` 或者为词根化的单词，或者为以上三种形式的列表

3. 从 $Q_{bool}$ 得到嵌套列表后，可以进行计算操作了，将列表传入``boolean_retrieval(search, inverted_table)``，其中``search``是输入的搜索列表，``inverted_table``是倒排索引表，这是一个递归函数，针对不同的输入，其输出分别如下：

   * 若输入为词根化的单词，则返回该单词对应的表项，即包含之的文章编号列表；
   * 若输入为形如``[exp,NOT]``的列表，则返回不在列表``boolean_retrieval(exp, inverted_table)``中的文章编号列表；
   * 若输入为形如``[exp1,exp2,AND]``的列表，则返回同时在列表``boolean_retrieval(exp1, inverted_table)``和``boolean_retrieval(exp2, inverted_table)``中的文章编号列表；
   * 若输入为形如``[exp1,exp2,OR]``的列表，则返回在列表``boolean_retrieval(exp1, inverted_table)``中或在``boolean_retrieval(exp2, inverted_table)``中的文章编号列表；

   最后的输出，就会是表达式``search``对应的文章编号列表了

4. 最后，我们根据得到的文章编号列表，到列表``file_name_list``中获得原文档的路径，并按照路径，输出各篇文章的标题。

### ``tf-idf``矩阵的计算

对于``tf-idf``矩阵的计算，我们的工作分为以下几个步骤：

1. 首先，必须要建立语料库，即从我们的文章集合中建立这样的语料库``corpus``，采用两级嵌套列表的形式来建立和存储：逐个访问预处理后的单词集合，将其分别添加到语料库中。这里为了避免调试繁琐，后续多次做无用功，选择将语料库存于硬盘中，其路径为 ``.\output\Edited_dataset\corpus.json``
2. 语料库建立后，我们针对每个单词，计算其``idf``值，并将其以字典形式存储于硬盘中，其路径为 ``.\output\Edited_dataset\word_idf_dict.json``
3. 在读入``corpus``和``word_idf_dict``之后，我们对于语料库中的每一篇文章，针对出现在其中的单词，分别计算其 ``tf-idf`` 值，我们将最终结果存储于一个二级字典``tf_idf_matrix``中，字典的一级键为文章编号，二级键为单词编号，值为对应的``tf-idf``值，这样存储是为了便于后期提取单独一篇文章的``tf-idf``向量。
4. 在遍历结束后，我们将矩阵字典存于硬盘中，其路径为``.\output\Edited_dataset\tf_idf_matrix.json``。其大小约为``1.3GB``

### 语义查询

对于语义查询，我们的工作大致分为以下几个步骤：

1. 首先打开单词和其编号对应的字典``word_dict``以及单词和其``idf``值对应的字典``word_idf_dict``
2. 接收输入字符串，并调用 ``word_tokenize`` 函数进行分词处理得到其单词列表。
3. 计算该字符串对应的稀疏``idf``向量，并转换为``scipy``包提供的``sparse.coo_matrix``形式。
4. 打开之前存储的``tf_idf_matrix``矩阵，遍历其中每个文章对应的稀疏``idf``向量，计算其与输入字符串对应向量的欧氏距离，这里调用了``numpy``中的``linalg.norm()``进行计算。同时，维护一个容量为``10``的列表，其中元素形式为二元组``(index, distance)``，``index``为文章编号，``distance``为该文章对应向量与输入向量的欧氏距离，保证列表中的元素为已经计算的文章中 ``distance`` 值最小的10个。

# 性能优化

整个实验过程中，大多数实验步骤的所耗时间都是在可以接受范围的，在实验过程中进行过性能优化的部分和优化过程如下：

### 倒排索引表的建立过程

* 问题：在初次建立倒排索引表时，我会针对每个出现在每篇文章中的单词，先判断该文章编号是否存在于单词对应的表项中，若不存在，再进行插入操作。但这样的操作，在当每个单词对应表项的列表数据巨大，当某一篇文章中某个单词重复出现时，其时间开销会非常巨大。测试证明，当检索的文章数目到达十万量级之后，操作的速度将会有巨大下降，由最初的每秒几百篇下降到每秒几篇，这种时间开销是我们无法接受的。
* 问题解决：我们发现，每个单词对应的表项，其中的文章编号序列，是递增的，而且我们的建立过程中，对于文章的访问，其序列也是递增的，那么只需要判断列表的最后一项与当前的文章序号是否相等即可。
* 优化结果：修改之后，建表的速度不会随着表项的增大而有明显下降，整个建表过程在一小时之内完成。

### ``tf-idf``矩阵的计算过程

* 问题：在最初，我们使用``nltk``包提供的工具计算``tf-idf``值，但测试发现，由于我们的语料库过于庞大，``nltk``包的计算速度是我们不可接受的（约每秒1个值）。
* 问题解决：我们将包的工作完全拆开，造轮子，自己编写``tf-idf``值的计算函数。编写了两个函数``tf(word, article)``和``creating_idf_dict(corpus, word_list)``，前者返回``word``和``artile``对应的``tf``值，后者将建立一个字典，存储了每个单词对应的``idf``值，并将其存储于硬盘中，路径为 ``.\output\Edited_dataset\word_idf_dict.json``

* 优化结果：使用自己构建的函数，``tf-idf``矩阵的计算时间下降到了2小时以内，是可以接受的

## 结果展示

### 布尔查询

在控制台中运行``python boolean_retrieval.py``，再输入查询字符串，可以得到结果，示例使用``hello AND world``，结果输出如下：

```
Matched file numbers:
[1707, 8041, 37786, 40265, 45727, 46211, 46487, 56269, 58670, 64029, 70356, 73291, 74085, 81849, 83162, 87527, 95075, 106339, 107024, 110128, 133027, 136456, 140589, 148829, 150472, 150657, 152226, 154849, 156984, 160420, 165336, 170855, 180843, 181899, 188928, 190393, 194614, 195346, 200227, 200687, 204517, 206055, 215089, 217638, 220835, 223591, 225860, 227465, 238211, 245118, 260300, 285506, 285575, 291932, 303425, 303985, 305383, 306103]
And their titles:
1707 : How this self-taught 14-year-old kid became an AI expert for IBM
8041 : Apple and Reese Witherspoon Are Developing Another Original TV Series - Fortune
37786 : Google's Waymo Testing Its Self-Driving Minivans In Atlanta | Fortune
40265 : CNBC Interview with Luxembourg’s Finance Minister Pierre Gramegna from the World Economic Forum 2018
45727 : Brainstorm Health: Trump at Davos, Monkey Clones, Azar HHS Confirmation | Fortune
46211 : CNBC Interview with OECD Secretary General, Angel Gurria, from the World Economic Forum 2018
46487 : How the World Economic Forum Brought #MeToo to Davos | Fortune
56269 : Daymond John tells shy airline passenger: You should have talked to me on the plane
58670 : CryptoKitties is Going Mobile. Can Ethereum Handle the Traffic?
64029 : SANRIO, Inc. Names Craig Takiguchi Chief Operating Officer
70356 : Intel's Former President Renee James Debuts Ampere Computing | Fortune
73291 : 2018 Winter Olympics: How to Watch on NBC, Streaming Online | Fortune
74085 : Brainstorm Health: New Zealand's HIV Prevention Plan, Apple Watch Diabetes Study, FDA Vs Kratom | Fortune
81849 : Tony Goncalves Appointed CEO Of Otter Media
83162 : North Korean leader's sister to visit South Korea for winter Olympics
87527 : Puma upbeat for 2018, sees turnaround for soccer sales
95075 : Brainstorm Health: Roche Buys Flatiron, FDA Alzheimer's Guidance, Flu Vaccine Effectiveness | Fortune
106339 : South Korea Crypto, Qualcomm, Russia Bot Army: Data Sheet, 20 Feb 2018
107024 : Otter Media Builds Senior Executive Team
110128 : Priceline Group to Change Name to Booking Holdings
133027 : Brainstorm Health: Health Care Data Danger, Big Pharma and Big Data, Ionis Huntington's Drug
136456 : Brainstorm Health: Hospital Bills, Novartis Virtual Trials, FDA Chief on Drug Prices
140589 : Designers Need to Start Talking About Bad Design | Fortune
148829 : Ivanka Trump, Elizabeth Warren, Rent the Runway: Broadsheet March 12
150472 : Latin American fans race to learn Russian before World Cup
150657 : Kayako Acquired by ESW Capital
152226 : CNBC EXCLUSIVE: CNBC TRANSCRIPT: SENIOR CONTRIBUTOR LARRY KUDLOW ON CNBC’S “CLOSING BELL” TODAY
154849 : Brainstorm Health: Freezing Your Brain, Lundbeck Parkinson's Deal, Louise Slaughter Passes | Fortune
156984 : Spotify IPO: A Look at the Unusual Route to Wall Street
160420 : Brainstorm Health: AI and Staffing, Virtual Reality Boost, Abortion Ban Block | Fortune
165336 : Brainstorm Health: Black Lung on the Rise, Digital Contraception, America's 'Healthiest' Communities
170855 : Brainstorm Health: Most Innovative Drug Makers, Israel's Health Data Push, Shire Stock Soars
180843 : Brainstorm Health: Boehner Joins Marijuana Company, Alzheimer’s Gene Deletion, Doctor Pay Gap
181899 : World's richest self-made woman shares 3 pieces of advice for success
188928 : RLH Corporation Enters Into Definitive Agreement to Acquire the Knights Inn Brand From Wyndham Hotel Group
190393 : Brainstorm Health: Jamie Dimon on Amazon/JPM/Berkshire, Opioid Treatment Costs, AbbVie Humira Deal | Fortune
194614 : Modavate Opens New Headquarters Facility in Buford, Georgia
195346 : CNBC Exclusive: CNBC Transcript: U.S. Treasury Secretary Steven Mnuchin on CNBC’s “Power Lunch” Today
200227 : Commentary: Why hating Facebook won't stop us from using it
200687 : COLUMN-Why hating Facebook won't stop us from using it
204517 : Brainstorm Health: Alzheimer's Definition, At-Home DNA Tests, UnitedHealth's Reach | Fortune
206055 : Contino Acquires Australian DevOps Leaders Nebulr to Accelerate Growth in APAC Region
215089 : Brainstorm Health: Digital Diabetes Prevention, Cannabis-Based Drug, UnitedHealth Earnings | Fortune
217638 : CNBC Interview with Tommy Koh, Ambassador-at-Large at Singapore’s Ministry of Foreign Affairs
220835 : Silicon Valley Poses a Challenge to Middle America
223591 : Silicon Valley Poses a Challenge to Middle America
225860 : #MeToo Comebacks, Matt Lauer, Charlie Rose: Broadsheet April 23
227465 : Brainstorm Health: Royal Baby Costs, VA Secretary Under Fire, Human DNA Structure | Fortune
238211 : CNBC First On: CNBC Transcript: National Economic Council Director Larry Kudlow on CNBC’S “Squawk on the Street” Today
245118 : Brainstorm Health: Ingestible Gut Bacteria Monitor, Athenahealth Pressure, Open Offices and Biopharma
260300 : How to start a profitable business with less than $500 in two weeks
285506 : Brainstorm Health: FDA Name and Shame, ASCO Abstracts, Non-Opioid Withdrawal Drug
285575 : Lamar Advertising Company Announces Cash Dividend on Common Stock
291932 : Rebooting food: Finding new ways to feed the future
303425 : Metal to Acquire Crumbs Technologies Inc.
303985 : Germany's Neuer will be first choice if he makes World Cup - Bierhoff
305383 : Soccer: Germany's Neuer will be first choice if he makes World Cup-Bierhoff
306103 : Brainstorm Health: Trump Drug Price Prediction, Sanofi’s Roseanne Subtweet, Healthy Life Expectancy
time used:
11.389686584472656
```

可以看到查询时间约``11s``，是可以接受的

查询``sale AND NOT company OR NOT bank AND trump``，结果过长，不便展示，查询时间约为`700s`，主要原因是`NOT`查询的时间较长

### 语义查询

我们所实现的语义查询的效率并不甚高，对于全体文档集合查询一次的耗时大约在半小时左右，且经过研究发现，性能的瓶颈在于计算两个稀疏向量的距离，这里采用的方法是``numpy.linalg.norm()``，其耗时大约为``7.6s/1000 article``，即计算1000次平均耗时约`7.6s`。这里测试前1000篇文章以作功能展示：

在控制台中运行``python boolean_retrieval.py``，再输入查询字符串，可以得到结果，示例使用``company percent income quarter``，结果输出如下（在前1000篇文章中最相关的10篇）：

```
699 : This was Jerry Seinfeld's first joke—and how he knew he'd found his calling as a comedian
637 : Cramer Remix: Boeing is a better bet than bitcoin
186 : Despite rising oil prices, one OPEC producer has announced new measures to protect foreign reserves
754 : California Residents Recount Harrowing Floods and Mudslides
835 : David Rosenberg: Fed taking on role of 'serial bubble blower' and it won't 'end well'
285 : GoPro CEO says he'd consider selling the company, but is still planning to be independent
76 : This investor is still bullish on tech stocks in China
228 : UPDATE 1-China's Didi Chuxing buys control of Brazil's 99 ride-hailing app
475 : Stocks making the biggest moves premarket: KODK, TGT, DE, SVU & more
147 : Last year's S&P 500 losers could be prime for speculation
```



