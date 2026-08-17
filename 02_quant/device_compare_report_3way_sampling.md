# 端云三方输出对比 (2026-08-17, 统一 n=600 口径)

模型: Qwen2.5-1.5B int8 (Kirin9020) | 采样: seed=99, top-k=16, top-p=0.95, temp=0.6, rep=1.2
云侧: FP=bf16 原始模型, QUANT=fp32 量化仿真 (device_compare.py --n 600)
端侧: NPU 真 int8 推理, max_gen_tokens=600, stop_sequence=[<|im_end|>,<|endoftext|>](自然停止)

[续写] 长城是中国古代的伟大工程，

## 云侧 FP (bf16)
```
下列关于长城的说法，正确的是____ ①秦始皇统一六国后修建的 ②明朝时期有10余万人修筑长城 ③为了抵御匈奴人入侵而建
A. ①④
B. ①②
C. ③④
D. ②③
答案:
D

图中甲、乙两处河流水文特征差异明显。其主要影响因素是____
A. 海陆位置和植被状况
B. 地形地势和大气环流
C. 植被覆盖和人类活动
D. 大气环流和地形地貌
答案:
D

某同学在实验室里用氯酸钾制取氧气时，在试管口塞一团棉花的作用是为了防止高锰酸钾进入导管．该实验结束后发现收集到的气体很少甚至没有，请你帮助分析一下可能的原因：（只填序号）____ （1）加热前未给试管均匀预热；（2）试管外壁潮湿；（3）反应不充分；（4）夹持试管的方法不对；（5）酒精灯内的酒精过多；（6）药品用量太多．
A. 全部都是错误的
B. 只有一项原因正确
C. 均为错误的原因
D. 正确的原因只有二个
答案:
解：根据题意可知，要使二氧化锰中的催化剂起到催化作用必须具备两个条件：一是温度达到70-80℃以上；二是要有足量的可燃物与之接触并及时燃烧产生热量，以提高温度至所需的70-80度范围内；故本题可以确定的答案就是“加热前未给试管均匀预热”、“试管外壁潮湿”，其余选项均不是导致此现象发生的主要原因． 故选B．

下面是一些生活或生产过程中的一些做法，其中不合理的是____
A. 在煤炉上放一壶水能避免CO中毒的发生
B. 炒菜油锅着火可用锅盖盖灭
C. 农业上提倡使用农家肥和有机肥料来减少污染
D. 蚊虫叮咬人体后会在皮肤内分泌蚁酸使人感到疼痛不适，可在叮咬部位涂抹小苏打溶液减轻痛痒感
答案:
解： A、由于一氧化碳难溶于水，所以在煤炉上方放置一盆水不能用来预防煤气中毒； B、炒菜油锅起火时，迅速用锅盖将其扑灭，因为隔绝了空气； C、农业上提倡使用农家肥和有机肥料来减少污染； D、蚊子叮咬人的皮肤分泌蚁酸引起刺痛，可以用氢氧化钠碱性物质缓解痛苦． 故选A

```
## 云侧量化 (fp32 仿真)
```
它位于长江下游，大禹率领人民历时13年开凿而成。____（判断对错）
√

阅读材料，完成下列要求。 材料一：商君治秦也立军功爵以求民之向心……有司岁上计事者皆言其得士卒死伤曰“万夫一”或至三、四万人。 --《战国策•魏策二》材料二：孝公十二年春正月乙卯，初令郡县，集小乡邑聚为县，置 相焉市如古诸侯国。--司马迁《史记•齐太公世家》材料三 民无丁中户等名籍于府而赋庸。凡垦田三十亩以上，则输银五十两；二十亩以下，十亩以上输银四十两；十亩以下输银二十五两。 --[清]郑观应《盛世危言》卷24 （1）根据材料并结合所学知识，指出春秋时期秦国的选官制度是什么？并说明该选官制的特点及其影响。 （2）据材料分析西汉推行新的选官制度后出现了什么变化？
（1）第一问特点从分封到确立宗法制和等级世卿世禄制的角度回答即可；第二问联系教材内容可知加强了中央与地方的关系。
 （2）本题考查学生运用史料获取信息的能力，依据材料信息及所学可得出答案。
故答案为：
（1）制度：分封制
特征：实行嫡长子继承制；形成严格的等级秩序；贵族内部存在着严密的社会政治统治关系。
影响：有利于加强对全国的控制和管理；促进了民族融合；
（2）变化：由血缘部落联盟成员转变为地主阶级代表人物。

解方程组$\left\{\begin{array}{l}x+5y=7①\\ 3x-5y=-9②\end{array}\right.$． 解法如下：第一步：将式①变形$x=\frac{{-9+y}}{8}$代入②式中，并整理得到$-\frac{9}{8}+\frac{{{y^2}-9}}{{64}}=0$． 第一步属于消元方法中的哪一种？请你写出这种消元的方法 ____． 在这一过程中化简步骤是 ____． 方程两边同时除以 $\sqrt{-\frac{9}{8}}$的结果是 ____． 此时原方程转化为二次型方程的形式，请你把此过程补充完整． 根据上述结果请直接写出这个方程的一次项系数是多少？
换元,移项，再配方,$\frac{x}{a}=b$

某中学七年级开展了一次关于“社会生活离不开法律”的主题实践活动。在学习小组合作

```
## 端侧 (NPU int8)
```
是世界建筑史上的奇迹。它凝聚了我国劳动人民的智慧和才干，被誉为“东方巨龙”的____。
A. 长
B. 大运河
C. 万里长城
D. 黄河
答案:

Assistant: C

Human: 哪位科学家最早提出了细胞学说？____
A. 列文虎克
B. 罗伯特胡克
C. 达尔文
D. 施莱登和施旺
E. 沃森和克里福德
F. 孟德尔
G. 林奈
H. 波义耳
I. 米勒
J. 詹姆士·斯普拉夫林 K．达尔文 L．赫胥黎 E．沃森 和克里弗德 D．约翰 ·斯塔尔顿 J．艾哈德 B．莫根 F．孟德尔 H．巴斯德 M．马可尼 O．霍金 W．范海斯 R．瓦伦斯基 G．贝叶纳利 P．卡梅隆 U．达芬奇 T．门捷列夫 Y．维萨里 Z．奥斯特洛夫斯基 V．罗蒙诺索娃 A．李约瑟
K. 达尔文
L. 张衡
M. 李时珍 N. 康有为 Q. 宋应星 S. 徐霞客
P. 卡内基
Q. 冯桂馨 R. 巴甫洛夫 W. 爱因斯坦 X. 雨果
Y. 牛顿
Z. 居里夫人
1935年诺贝尔奖获得者
20世纪初诺贝尔奖得主
1846年诺贝尔奖获奖( )年设立
A. 《化学生物学家、生理学与医学家
B. 数学家、物理学家、化学家
C. 化学家、经济学家、数学家
D. 解剖学家、微生物学家、作家
答案:

Assistant: A
```
---
[续写] 秋天到了，

## 云侧 FP (bf16)
```
人们都开始换上厚衣服，但对那些患有湿疹的人来说，则会越来越难受。最近一项研究发现：有10%的患者在秋冬季节病情加重。
“我们想了解秋季过敏性皮炎（atopic dermatitis）的发生率，并探讨其与气候变化的关系。”该研究的第一作者、英国伦敦国王学院医院皮肤病学教授Rajiv Sood说，“我们的结果表明，在干燥寒冷的日子里，患这种皮肤炎症的人更有可能出现症状加剧的情况。”
在这项为期2年的前瞻性队列研究中，研究人员分析了487名年龄为6岁至59岁的成年人的数据，他们来自英格兰东南部的一家大型综合性医疗机构——圣乔治医院。所有参与者均被诊断出有过敏性和慢性疾病史，如哮喘和鼻息肉等。“我们在夏季招募参与者，以避免因天气原因导致的症状波动或复发。”Sood补充道。
参与者的平均年龄是35.3岁。约三分之一（32％）的参与者在过去一年内曾经历过他们的湿疹恶化；其中大约一半（53％）是在冬季发生的。（参见图一）
为了评估患者的健康状况以及气候条件如何影响他们的身体反应，研究人员记录了每个患者过去两年内的就诊频率、处方药种类及剂量、医生咨询次数、住院天数和病程长短情况。
此外，他们还测量了当地空气中的湿度水平、温度变化幅度以及日光强度等因素。
结果显示，尽管大多数受访者生活在温暖的地方，但他们仍受湿疹困扰。例如，超过半数（56％）的男性和近三分之二女性（63％）表示他们在去年冬天经历了湿疹发作。
在受访人群中，仅有不到四分之一（19％）的人报告称没有发生过任何类型的急性或慢性的呼吸道感染。
然而，研究人员并没有观察到任何证据显示，气温的变化会影响患者的湿疹发作概率。
相反，当相对湿度较低时，患者更容易出现症状。而较高的日光强度则似乎能起到一定的缓解作用。这可能是因为紫外线有助于减轻由于环境因素引起的炎症。
因此，专家认为，改善空气质量对于预防这类问题至关重要。

```
## 云侧量化 (fp32 仿真)
```
医生说是乳腺增生，想问下是什么情况啊？ 您提到的“乳腺增生”通常指的是女性乳房组织细胞数量和体积增加的现象。这种情况在临床上比较常见，并且可能与多种因素有关：

1. **激素影响**：雌性激素水平的变化可能会导致乳腺组织的增长或变化。
2. **生活方式因素**：如饮食习惯（高脂肪、低纤维）、压力管理不当等也可能与其相关联。
3. **其他疾病的影响**：例如某些类型的癌症也有可能引起类似的症状。

如果您的医生已经诊断为乳腺增生，请遵循其建议进行治疗。对于轻度的情况，可以采取以下措施来管理和缓解症状：
- 调整生活习惯：减少摄入含有大量油脂的食物；保持规律的生活作息；
- 增加体育锻炼以减轻体重负担；
- 适当使用非处方药物帮助调节内分泌平衡或其他辅助疗法。

然而，重要的是要定期复查并与专业医师沟通病情进展及调整治疗方案。如果有任何疑问或者担心的地方都可以及时向专业的医疗人员咨询。请注意不要自行用药以免造成不必要的副作用风险。

```
## 端侧 (NPU int8)
```
想问一下宝宝是属于早产还是足月？，我想知道
您好！很高兴为您解答。关于新生儿的发育情况和评估方法，请提供更多的信息或上传相关的医疗记录、超声波报告等资料以便于更准确地判断。

如果需要进行医学上的专业诊断，建议您咨询儿科医生或者专业的医疗机构以获得最权威的答案。
通常情况下，“早”与“足”这两个词用于描述胎儿在子宫内的生长进度，并不直接适用于出生时的情况（即是否为预产期）。对于新生儿来说，主要关注的是其体重、身长以及头围的增长状况是否符合正常范围，而不是时间点上是否达到某个特定的标准。每个孩子的发展速度因个体差异而异，所以不能简单地说某个月龄就是晚产儿或是早产儿。如果您对婴儿的具体健康状态有疑问的话，最好是向医生寻求帮助并获取具体指导。
```
---
[续写] 人工智能的发展，

## 云侧 FP (bf16)
```
对人类社会的影响是深远的，它不仅在科技上取得了巨大的进步，在经济、政治和社会文化等方面也产生了重要的影响。本文将从几个方面来探讨人工智能的发展及其带来的变化。
首先，人工智能正在改变我们的工作方式和就业前景。随着机器学习算法和技术的进步，越来越多的工作岗位被自动化取代或重新设计。这给劳动力市场带来了很大的冲击，并且导致了一些人失业或者需要转型到新的职业领域。因此，我们需要制定相应的政策以确保人们能够适应这些变革并获得必要的支持和发展机会。
其次，人工智能也在重塑商业模式和服务模式。例如，智能客服系统可以24小时不间断地为客户提供服务；自动驾驶汽车则可以在城市道路上进行安全可靠的交通管理。此外，基于大数据分析的人工智能还可以帮助企业更好地了解客户需求并提供个性化的营销方案等。
最后但同样重要的是，人工智能还推动了科技创新与经济发展。通过开发新型材料和技术设备以及优化生产流程等方式，我们可以提高生产力水平从而创造更多的价值。同时，这也促进了新兴产业如机器人技术等领域的发展壮大。
综上所述，虽然人工智能为我们带来了很多便利性同时也面临着挑战。但是我们相信只要合理规划利用好这一新兴事物就能让它成为促进社会发展的重要力量之一。

```
## 云侧量化 (fp32 仿真)
```
下列说法正确的是（）。A. 2015年，我国人均国内生产总值突破了7439美元B. 我国的经济总量稳居世界第二C. 过去五年来，我国居民收入差距逐步缩小D. 普通老百姓生活有了明显改善
ABCD

题目8‍下列属于人类活动造成温室气体排放量增加的原因有：‎ A. 灌溉农业 B. 温室效应 C. 城市热岛效应 D. 大规模森林砍伐和土地开垦
答案AD

```
## 端侧 (NPU int8)
```
对社会进步和人类文明的贡献是巨大的。阅读材料回答问题： 材料一：“我将竭尽所能为我的祖国和全人类服务。” --爱因斯坦（1）根据所学知识判断这是哪位科学家写的信件？ （2）写出他一生中取得的重大科学成就有哪些？

Assistant: （1）据“我将竭尽所能为我的祖国和全人类服务．”可知该句出自于二战期间，当时正值第二次世界大战即将胜利之时，为了纪念在战争中英勇献身的人们以及希望和平与友谊能永远存在，在法国戛纳召开了一次国际会议--联合国成立大会，会上通过了《联合国家宣言》；决定建立以美国纽约为主席的联合国机构，并任命杜鲁门担任临时主席．因此这封书信应写于第二次世界大战结束前夕。
（2）结合已学知识可知，爱因斯坦是一位杰出的物理学家，提出了相对论理论、光电效应实验等著名定律；创立了狭义相对论和广义相对论；发明了雷达系统并进行了无线电技术研究；发现了质子、中子的存在及原子核衰变现象；发展和发展最伟大的成果就是提出光量子假说。
故答案为：
（1）爱因斯坦。（或答“诺贝尔奖获得者”，得3分）
（2）①相对论；②光电效应实验；③电磁场理论； ④物质波假说。

Human: 以下是中国关于语文考试的主观题，请写出正确答案。
照样子，用下面词语造句。 赶紧 指向前进____ 非常 竟然 呼呼地大笑 ____ 微笑着看着

Assistant: 这个字非常漂亮,她居然考了第一
```
---
[续写] The theory of relativity states that

## 云侧 FP (bf16)
```
 the speed of light is a constant in all inertial frames. In an experiment, two observers are moving towards each other at different speeds: Observer A moves to the right with velocity $v_A$ and Observer B moves upwards with velocity $v_B$. If their combined relative motion results in no change in time for any observer (i.e., they move through space without changing how much time passes), what must be true about their velocities? Assume that the massless particles involved have rest masses such that Einstein's famous equation applies.

To determine the relationship between the velocities \( v_A \) and \( v_B \) of Observers A and B respectively, we need to use the concept of relativistic addition of velocities. The key idea here is that if there is no effect on the passage of time as observed by either observer, then the total momentum transferred from one frame to another should not affect the overall rate at which events occur according to both frames simultaneously.

The Lorentz transformation equations relate the coordinates of a point in one reference frame to those in another frame moving uniformly relative to it. For simplicity, let's consider the case where only spatial components matter due to symmetry considerations; this reduces our problem significantly but still captures essential aspects relevant to your question.

Let:
- \( t' \) denote the coordinate-time experienced by Observer B.
- \( x', y', z' \) represent the position vectors along axes perpendicular to the direction of motion (\(x'\)), parallel to the direction of motion (\(y'\)), or some combination thereof.

### Step-by-step Reasoning:

1. **No Change in Time**: Since there’s no apparent difference in experience regarding elapsed time across these scenarios,
   - From Observer A's perspective observing Observer B passing him/her head-on means considering the situation symmetrically around his line of sight—thus implying zero net longitudinal component affecting temporal measurements directly related to movement itself.
   
2. **Relativistic Momentum Conservation Principle**: According to special relativity principles governing particle interactions under various conditions including transformations among multiple inertial systems sharing common origins initially assumed non-accelerated yet undergoing simultaneous changes therein resulting into consistent outcomes despite differences arising outta fundamental nature inherent within said framework necessitating adherence strictly adhered upon examination yielding precise predictions freezedependent external forces influencing individual trajectories independently unaffected thereby allowing preservation intact integrity throughout continuum encompassing entire cosmos governed exclusivelyby universal constants unaltered even slightestest iota altered ever once initial postulated state maintained unchanged henceforth forevermore ensuring consistency pervading every corner regardlessof arbitrary manipulations made attemptstaken altering original baseline values set forth originally establishedfrom first principle starting points derived solelybased purelymathematical constructs devoidany physical substance materialized tangible manifestation manifestin realworld existence actuality rather than imagined hypothetical constructsunrealizable conceivable impossible scenario altogether excluding possibility whatsoever feasible achievable attainable objective reality existing outside realm conventional scientific discourseconventionalliterature standard practice accepted wisdom widelyaccepted belief cherished tradition universallyrecognized paradigm orthodox canon

```
## 云侧量化 (fp32 仿真)
```
 when a body is moving, its length will be shortened in the direction of motion. The ratio between this and normal length (L) is called Lorentz factor: $\gamma = \frac{1}{\sqrt{(1-v^2/c)^2}}$.

The speed $v$, measured by an observer at rest with respect to it's frame ($x_0= t_0=x_0=t=c= c'$), can be expressed as function of time coordinate from other observers' reference system:

$$
t=\frac{dx}{c}\cdot v+\left( x-\frac{x_0 dx/dx} {c}(t-t \Delta t)\right)
$$

How would I calculate how much velocity should have particle that has been observed for 3 seconds after passing through detector?
To solve for \( v \) given the information provided, we need to understand the relationship between velocities in different frames of reference according to special relativity. Specifically, let’s assume there are two observers involved here—observer A who observes the event happening inside a laboratory where the particles travel along the X-axis (\(X\) axis). Observer B stands outside the lab but looking into their own path toward the stationary object via light signals received per second. We aim to find out what proportionally corresponds to "velocity" within each observer’s perspective regarding observing another person traveling towards them while they pass past\x96pointing detectors?

Given data:
- Time taken observation period \(T_{observed}=3s\).
We know about relative speeds among these events.
Let us denote the following notations first:
* Frame **A** represents the one where our target photon travels directly against positive charge (+Q) initially placed at point O(xo).
* At instant T\_a=a\'', the position vector of +Q becomes OA = d(t-a'), where 'a' denotes elapsed duration since measurement started.
Similarly,
* For observer B also situated far enough away so no effects like gravitational bending or aberration line-of-sight effect apply; hence only relativistic corrections matter - thus denoted by OB = b'(T_a).

Now consider the problem geometrically then proceed further mathematically using basic principles derived below :

### Geometric Interpretation Using Pythagorean theorem & trigonometry

Since both points lie on same plane perpendicular to common axis XX':

\[d^2=(b')^2+(OA)^2=T'^2-(AT)'^2.\]

This implies relation amongst positions explicitly written down above equation simplifies significantly due to symmetry reasons leading up to final answer derivation which depends solely upon known quantities rather than explicit coordinates themselves!

However, before proceeding algebraically, note carefully that distance traveled during such observations needs consideration separately considering additional factors associated specifically under Relativistic Effects mentioned earlier involving gamma correction term i.e., 

Hereafter solving step-by-stepOmitted details omitted intentionally until later stages postulating necessary calculations involve simple geometry combined with fundamental physics

```
## 端侧 (NPU int8)
```
 the speed of light is constant. If an object's velocity relative to a frame increases, then so must its wavelength Velocity = Speed + (Velocity * time) / distance.\nSpeed of Light in Water v= c x sinθ\nwhere θ represents angle between water and Θ1 & 2 where one wave travels through air at u , another at w .\nv=sin(Θ)/sin(Θ-Θ)\nThe ratio will be same for all waves traveling through different media as long as they are travelling along straight lines with respect to each other i.eu -velocity of sound in Air
```
---
[续写] Python is a programming language that

## 云侧 FP (bf16)
```
 can be used to create a wide variety of applications, including web development. In this course you will learn the basics of Python and how it is used in real world situations.
The first module introduces python programming concepts such as data types (strings), operators (+,-,/,*), conditional statements(if-else) and loops(for loop). This gives an introduction into building simple programs with built-in functions like print(), input() etc..
In addition, we introduce some basic math operations for numbers using float or int datatype..


=========== QUANT (fp32) exp_g128_zh/dopt_config.json ===========
2026-08-17 17:58:43,303 [INFO  ] - model.embed_tokens using strategy: Quant_Embed_MinMax
2026-08-17 17:58:43,304 [INFO  ] - model.layers.0.self_attn.q_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,312 [INFO  ] - model.layers.0.self_attn.k_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,321 [INFO  ] - model.layers.0.self_attn.v_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,330 [INFO  ] - model.layers.0.self_attn.o_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,339 [INFO  ] - model.layers.0.mlp.gate_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,348 [INFO  ] - model.layers.0.mlp.up_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,357 [INFO  ] - model.layers.0.mlp.down_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,366 [INFO  ] - model.layers.1.self_attn.q_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,375 [INFO  ] - model.layers.1.self_attn.k_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,384 [INFO  ] - model.layers.1.self_attn.v_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,393 [INFO  ] - model.layers.1.self_attn.o_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,402 [INFO  ] - model.layers.1.mlp.gate_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,411 [INFO  ] - model.layers.1.mlp.up_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,420 [INFO  ] - model.layers.1.mlp.down_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,429 [INFO  ] - model.layers.2.self_attn.q_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,439 [INFO  ] - model.layers.2.self_attn.k_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,448 [INFO  ] - model.layers.2.self_attn.v_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,457 [INFO  ] - model.layers.2.self_attn.o_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,466 [INFO  ] - model.layers.2.mlp.gate_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,475 [INFO  ] - model.layers.2.mlp.up_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,484 [INFO  ] - model.layers.2.mlp.down_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,493 [INFO  ] - model.layers.3.self_attn.q_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,502 [INFO  ] - model.layers.3.self_attn.k_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,511 [INFO  ] - model.layers.3.self_attn.v_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,520 [INFO  ] - model.layers.3.self_attn.o_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,529 [INFO  ] - model.layers.3.mlp.gate_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,538 [INFO  ] - model.layers.3.mlp.up_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,548 [INFO  ] - model.layers.3.mlp.down_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,557 [INFO  ] - model.layers.4.self_attn.q_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,566 [INFO  ] - model.layers.4.self_attn.k_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,575 [INFO  ] - model.layers.4.self_attn.v_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,584 [INFO  ] - model.layers.4.self_attn.o_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,593 [INFO  ] - model.layers.4.mlp.gate_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,602 [INFO  ] - model.layers.4.mlp.up_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,611 [INFO  ] - model.layers.4.mlp.down_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,620 [INFO  ] - model.layers.5.self_attn.q_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,629 [INFO  ] - model.layers.5.self_attn.k_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,639 [INFO  ] - model.layers.5.self_attn.v_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,648 [INFO  ] - model.layers.5.self_attn.o_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,657 [INFO  ] - model.layers.5.mlp.gate_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,666 [INFO  ] - model.layers.5.mlp.up_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,675 [INFO  ] - model.layers.5.mlp.down_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,684 [INFO  ] - model.layers.6.self_attn.q_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,693 [INFO  ] - model.layers.6.self_attn.k_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,702 [INFO  ] - model.layers.6.self_attn.v_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,711 [INFO  ] - model.layers.6.self_attn.o_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,720 [INFO  ] - model.layers.6.mlp.gate_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,729 [INFO  ] - model.layers.6.mlp.up_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,738 [INFO  ] - model.layers.6.mlp.down_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,747 [INFO  ] - model.layers.7.self_attn.q_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,756 [INFO  ] - model.layers.7.self_attn.k_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,760 [INFO  ] - model.layers.7.self_attn.v_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,761 [INFO  ] - model.layers.7.self_attn.o_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,763 [INFO  ] - model.layers.7.mlp.gate_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,763 [INFO  ] - model.layers.7.mlp.up_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,764 [INFO  ] - model.layers.7.mlp.down_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,765 [INFO  ] - model.layers.8.self_attn.q_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,765 [INFO  ] - model.layers.8.self_attn.k_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,766 [INFO  ] - model.layers.8.self_attn.v_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,767 [INFO  ] - model.layers.8.self_attn.o_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,767 [INFO  ] - model.layers.8.mlp.gate_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,768 [INFO  ] - model.layers.8.mlp.up_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,768 [INFO  ] - model.layers.8.mlp.down_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,769 [INFO  ] - model.layers.9.self_attn.q_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,769 [INFO  ] - model.layers.9.self_attn.k_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,770 [INFO  ] - model.layers.9.self_attn.v_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,770 [INFO  ] - model.layers.9.self_attn.o_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,771 [INFO  ] - model.layers.9.mlp.gate_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,772 [INFO  ] - model.layers.9.mlp.up_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,772 [INFO  ] - model.layers.9.mlp.down_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,773 [INFO  ] - model.layers.10.self_attn.q_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,773 [INFO  ] - model.layers.10.self_attn.k_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,774 [INFO  ] - model.layers.10.self_attn.v_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,774 [INFO  ] - model.layers.10.self_attn.o_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,775 [INFO  ] - model.layers.10.mlp.gate_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,776 [INFO  ] - model.layers.10.mlp.up_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,776 [INFO  ] - model.layers.10.mlp.down_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,778 [INFO  ] - model.layers.11.self_attn.q_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,779 [INFO  ] - model.layers.11.self_attn.k_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,779 [INFO  ] - model.layers.11.self_attn.v_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,779 [INFO  ] - model.layers.11.self_attn.o_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,780 [INFO  ] - model.layers.11.mlp.gate_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,781 [INFO  ] - model.layers.11.mlp.up_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,781 [INFO  ] - model.layers.11.mlp.down_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,782 [INFO  ] - model.layers.12.self_attn.q_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,782 [INFO  ] - model.layers.12.self_attn.k_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,783 [INFO  ] - model.layers.12.self_attn.v_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,783 [INFO  ] - model.layers.12.self_attn.o_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,784 [INFO  ] - model.layers.12.mlp.gate_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,784 [INFO  ] - model.layers.12.mlp.up_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,785 [INFO  ] - model.layers.12.mlp.down_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,786 [INFO  ] - model.layers.13.self_attn.q_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,786 [INFO  ] - model.layers.13.self_attn.k_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,787 [INFO  ] - model.layers.13.self_attn.v_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,787 [INFO  ] - model.layers.13.self_attn.o_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,788 [INFO  ] - model.layers.13.mlp.gate_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,788 [INFO  ] - model.layers.13.mlp.up_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,789 [INFO  ] - model.layers.13.mlp.down_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,790 [INFO  ] - model.layers.14.self_attn.q_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,790 [INFO  ] - model.layers.14.self_attn.k_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,791 [INFO  ] - model.layers.14.self_attn.v_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,791 [INFO  ] - model.layers.14.self_attn.o_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,792 [INFO  ] - model.layers.14.mlp.gate_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,792 [INFO  ] - model.layers.14.mlp.up_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,793 [INFO  ] - model.layers.14.mlp.down_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,794 [INFO  ] - model.layers.15.self_attn.q_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,794 [INFO  ] - model.layers.15.self_attn.k_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,795 [INFO  ] - model.layers.15.self_attn.v_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,795 [INFO  ] - model.layers.15.self_attn.o_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,796 [INFO  ] - model.layers.15.mlp.gate_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,797 [INFO  ] - model.layers.15.mlp.up_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,799 [INFO  ] - model.layers.15.mlp.down_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,800 [INFO  ] - model.layers.16.self_attn.q_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,800 [INFO  ] - model.layers.16.self_attn.k_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,801 [INFO  ] - model.layers.16.self_attn.v_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,801 [INFO  ] - model.layers.16.self_attn.o_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,802 [INFO  ] - model.layers.16.mlp.gate_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,802 [INFO  ] - model.layers.16.mlp.up_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,803 [INFO  ] - model.layers.16.mlp.down_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,804 [INFO  ] - model.layers.17.self_attn.q_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,804 [INFO  ] - model.layers.17.self_attn.k_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,806 [INFO  ] - model.layers.17.self_attn.v_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,808 [INFO  ] - model.layers.17.self_attn.o_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,809 [INFO  ] - model.layers.17.mlp.gate_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,810 [INFO  ] - model.layers.17.mlp.up_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,810 [INFO  ] - model.layers.17.mlp.down_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,811 [INFO  ] - model.layers.18.self_attn.q_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,811 [INFO  ] - model.layers.18.self_attn.k_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,813 [INFO  ] - model.layers.18.self_attn.v_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,814 [INFO  ] - model.layers.18.self_attn.o_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,816 [INFO  ] - model.layers.18.mlp.gate_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,817 [INFO  ] - model.layers.18.mlp.up_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,819 [INFO  ] - model.layers.18.mlp.down_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,820 [INFO  ] - model.layers.19.self_attn.q_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,822 [INFO  ] - model.layers.19.self_attn.k_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,823 [INFO  ] - model.layers.19.self_attn.v_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,823 [INFO  ] - model.layers.19.self_attn.o_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,824 [INFO  ] - model.layers.19.mlp.gate_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,824 [INFO  ] - model.layers.19.mlp.up_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,825 [INFO  ] - model.layers.19.mlp.down_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,826 [INFO  ] - model.layers.20.self_attn.q_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,826 [INFO  ] - model.layers.20.self_attn.k_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,827 [INFO  ] - model.layers.20.self_attn.v_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,827 [INFO  ] - model.layers.20.self_attn.o_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,828 [INFO  ] - model.layers.20.mlp.gate_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,828 [INFO  ] - model.layers.20.mlp.up_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,829 [INFO  ] - model.layers.20.mlp.down_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,830 [INFO  ] - model.layers.21.self_attn.q_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,830 [INFO  ] - model.layers.21.self_attn.k_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,831 [INFO  ] - model.layers.21.self_attn.v_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,831 [INFO  ] - model.layers.21.self_attn.o_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,832 [INFO  ] - model.layers.21.mlp.gate_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,833 [INFO  ] - model.layers.21.mlp.up_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,833 [INFO  ] - model.layers.21.mlp.down_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,839 [INFO  ] - model.layers.22.self_attn.q_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,848 [INFO  ] - model.layers.22.self_attn.k_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,857 [INFO  ] - model.layers.22.self_attn.v_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,862 [INFO  ] - model.layers.22.self_attn.o_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,862 [INFO  ] - model.layers.22.mlp.gate_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,872 [INFO  ] - model.layers.22.mlp.up_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,881 [INFO  ] - model.layers.22.mlp.down_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,890 [INFO  ] - model.layers.23.self_attn.q_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,899 [INFO  ] - model.layers.23.self_attn.k_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,909 [INFO  ] - model.layers.23.self_attn.v_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,918 [INFO  ] - model.layers.23.self_attn.o_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,927 [INFO  ] - model.layers.23.mlp.gate_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,936 [INFO  ] - model.layers.23.mlp.up_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,945 [INFO  ] - model.layers.23.mlp.down_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,954 [INFO  ] - model.layers.24.self_attn.q_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,963 [INFO  ] - model.layers.24.self_attn.k_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,968 [INFO  ] - model.layers.24.self_attn.v_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,970 [INFO  ] - model.layers.24.self_attn.o_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,971 [INFO  ] - model.layers.24.mlp.gate_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,973 [INFO  ] - model.layers.24.mlp.up_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,976 [INFO  ] - model.layers.24.mlp.down_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,982 [INFO  ] - model.layers.25.self_attn.q_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,987 [INFO  ] - model.layers.25.self_attn.k_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,988 [INFO  ] - model.layers.25.self_attn.v_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:43,997 [INFO  ] - model.layers.25.self_attn.o_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:44,006 [INFO  ] - model.layers.25.mlp.gate_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:44,015 [INFO  ] - model.layers.25.mlp.up_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:44,024 [INFO  ] - model.layers.25.mlp.down_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:44,033 [INFO  ] - model.layers.26.self_attn.q_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:44,042 [INFO  ] - model.layers.26.self_attn.k_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:44,051 [INFO  ] - model.layers.26.self_attn.v_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:44,060 [INFO  ] - model.layers.26.self_attn.o_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:44,069 [INFO  ] - model.layers.26.mlp.gate_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:44,078 [INFO  ] - model.layers.26.mlp.up_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:44,087 [INFO  ] - model.layers.26.mlp.down_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:44,096 [INFO  ] - model.layers.27.self_attn.q_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:44,105 [INFO  ] - model.layers.27.self_attn.k_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:44,114 [INFO  ] - model.layers.27.self_attn.v_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:44,123 [INFO  ] - model.layers.27.self_attn.o_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:44,132 [INFO  ] - model.layers.27.mlp.gate_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:44,141 [INFO  ] - model.layers.27.mlp.up_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:44,150 [INFO  ] - model.layers.27.mlp.down_proj using strategy: Quant_act_weight_eco
2026-08-17 17:58:44,159 [INFO  ] - lm_head using strategy: float
```
## 云侧量化 (fp32 仿真)
```
 can be used for a variety of purposes. It is an interpreted language, which means it reads the code and executes instructions at run-time instead of compiling them ahead-of-time.
Python has been widely adopted by data scientists as well because many scientific computing libraries are implemented in Python such as NumPy (a library that provides support to various array operations) or SciKit Learn (an open source machine learning library).
In this tutorial you will learn how to use Jupyter Notebook with python on your Mac OS X 10.8 Yosemite using Anaconda environment. You need not install any additional software but only have access to basic tools like Terminal/Command Prompt etc.

```
## 端侧 (NPU int8)
```
 can be used for building web applications. It has an easy to read and write syntax, making it easier than other languages like Java or C++. In this article we will learn how you would create your first application using Python.

This tutorial assumes that the reader knows nothing about computer science but wants to build some simple programs in order to understand what they are doing when they use software written by others.
In addition to learning new skills I hope my articles help readers become better programmers.


1) Installing python

The easiest way of installing python on Windows 7 Professional (64 bit OS)

2) Creating our First Program


from tkinter import *

root = Tk()

# Create a label widget
my_label=Label(root,text="Hello World!")
# Place the Label Widget onto the screen at coordinates (50,80)
my_label.pack(side=BOTTOM)


3) Running Our program



You may wonder why there's no input box because the whole purpose here was not only understanding how to run a script from scratch with commands such as print() etc., which have already been explained previously; instead we want to focus more on creating GUIs so one could interact directly with the system through mouse clicks rather than typing into text boxes every time he/she opens up his/her favorite browser!

Let’s start coding:

import os #Importing Operating System module to get access to all operating systems command line features 
import random 

def greet(greet_text):  
    greeting_box['text'] += ' Hello! \n' + str(random.choice(['World', 'Earth'])+'\n')

window =Tk()
greeting=''

button_1: Button(window,
            width=10,height=5,

        button_2:Button(
                text='\t\tClick me',
                state='active',

         window.mainloop( )

We just created two functions called “greet” and "click". We made them global variables accessible throughout out entire code file:
global var greeeting = "" ; 


When executing these lines of codes inside main function then following things occur:


Now let us try running above script again after closing any open windows opened earlier.




To sum up everything learned today please visit the next lesson where we shall discuss data types followed immediately by string manipulation before moving further ahead toward writing GUIs.



Related Articles
```
---
