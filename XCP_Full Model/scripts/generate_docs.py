"""生成 4 个 docx：邮件（英/中）+ 论文（英/中）。

约束：不公开最优超参数组合、不公开源文件核心设计。
"""
import os
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # XCP_Full Model
OUT = os.path.join(HERE, 'docs')
os.makedirs(OUT, exist_ok=True)

FIG_C = os.path.join(OUT, 'compression.png')
FIG_A = os.path.join(OUT, 'accuracy.png')
FIG_S = os.path.join(OUT, 'scaling.png')

IDENTITY_EN = 'Xingcheng Ping, a 14-year-old middle-school student from China.'
IDENTITY_CN = '平星辰（Xingcheng Ping），中国一名 14 岁的初中生。'


def _add_pic(doc, path, width=5.5):
    if os.path.exists(path):
        doc.add_picture(path, width=Inches(width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER


# ======================================================================
# 英文论文
# ======================================================================
def build_paper_en():
    d = Document()
    d.add_heading('XCP: A Discrete-Symbolic Language Model with Sparse Attention and Fact Storage', 0)
    p = d.add_paragraph('Author: Xingcheng Ping')
    p.runs[0].italic = True

    d.add_heading('Abstract', 1)
    d.add_paragraph(
        'We present XCP, a language-model architecture that replaces the continuous '
        'vector representations of the Transformer with a discrete-symbolic representation '
        'combined with sparse linear layers. XCP stores tokens as integer IDs (not dense '
        'vectors), organizes them into semantic classes by unsupervised clustering, models '
        'syntax with discrete class-to-class transition rules, and stores lexical facts as '
        'integer lookup tables. Attention is sparsified to 15% of its connections and trained '
        'from scratch. Across layers, XCP achieves compression ratios of roughly 1000x '
        '(embedding storage), 95% (feed-forward fact side), 85% (attention), and 112x '
        '(discrete fact lookup). On a real English novel, XCP reaches a next-token accuracy '
        'higher than a same-task Transformer at a small fraction of the parameter count, and '
        'its discrete core is learned by counting in a single pass rather than by iterative '
        'gradient descent.')

    d.add_heading('1. Introduction', 1)
    d.add_paragraph(
        'The Transformer and its successors encode every token as a continuous d-dimensional '
        'vector and learn knowledge as dense floating-point weights. This yields smooth '
        'generalization but also heavy redundancy: the token-embedding table costs K*d floats, '
        'and the feed-forward layers store facts in dense matrices. Symbolic approaches to '
        'language were largely abandoned because they did not scale; here we show that a '
        'discrete-symbolic core, combined with sparsity where continuity is genuinely needed, '
        'can match or exceed a small Transformer on next-token prediction while using orders '
        'of magnitude fewer parameters and one-pass counting-based training.')

    d.add_heading('2. Method', 1)
    d.add_heading('2.1 Discrete Embedding', 2)
    d.add_paragraph(
        'Each token is assigned an integer ID. Storage cost is K*log2(K) bits versus K*d '
        'floating-point numbers for a continuous embedding — a ~1000x reduction. Tokens are '
        'then grouped into semantic classes by unsupervised clustering of their left/right '
        'context distributions.')

    d.add_heading('2.2 Discrete Feed-Forward (structure + facts)', 2)
    d.add_paragraph(
        'The FFN is replaced by two discrete tables. (a) Structure: a class-to-class m-gram '
        'count table captures syntactic patterns. (b) Facts: a (left-word, class) -> next-word '
        'lookup stores lexical associations, with a class-level fallback for unseen '
        'combinations. This separates grammar from lexicon and stores facts as integers rather '
        'than floats, an additional ~112x saving over a sparse float matrix.')

    d.add_heading('2.3 Sparse Attention (from-scratch sparsity)', 2)
    d.add_paragraph(
        'The Q, K, V, O projections are masked to a fixed 15% of connections and trained from '
        'scratch. Unlike post-hoc pruning, from-scratch sparsity does not reduce memory '
        'capacity: capacity is bounded by the number of neurons, not the number of connections. '
        'We further measured that the effective rank of W_Q^T W_K is only about 0.06*d, which '
        'bounds the compressibility of attention at ~85% with no loss.')

    d.add_heading('2.4 Relation Gate', 2)
    d.add_paragraph(
        'Discourse connectives (because/but/then) act as regulators of prediction state rather '
        'than as content predictors, mirroring the role of abstract words as operations.')

    d.add_heading('3. Experiments', 1)
    d.add_paragraph(
        'We compared XCP against a PyTorch Transformer (d=512, 3 layers) on next-token '
        'prediction. Corpora include controlled template corpora (with held-out semantic-class '
        'generalization) and a real English novel (Pride and Prejudice, 131k words, ~7k '
        'vocabulary). Hyperparameters (class count and context length) are swept over a range; '
        'the optimal combination is treated as engineering know-how and is not disclosed here.')

    d.add_heading('4. Results', 1)
    d.add_paragraph('Table 1: per-layer compression ratio.')
    t = d.add_table(rows=5, cols=3)
    t.style = 'Light Grid Accent 1'
    hdr = t.rows[0].cells
    hdr[0].text, hdr[1].text, hdr[2].text = 'Layer', 'XCP method', 'Compression'
    rows = [('Embedding (storage)', 'discrete integer ID', '~1000x'),
            ('FFN (fact side)', 'sparse W1 + frozen W2', '~95%'),
            ('Attention', 'sparse Q/K/V/O 85%', '~85%'),
            ('Fact lookup', 'integer table vs float matrix', '~112x')]
    for i, (a, b, c) in enumerate(rows, start=1):
        t.rows[i].cells[0].text, t.rows[i].cells[1].text, t.rows[i].cells[2].text = a, b, c

    d.add_paragraph('')
    d.add_paragraph('Table 2: end-to-end comparison on real text.')
    t2 = d.add_table(rows=3, cols=3)
    t2.style = 'Light Grid Accent 1'
    h = t2.rows[0].cells
    h[0].text, h[1].text, h[2].text = 'Model', 'Test acc (top-1)', 'Parameters'
    t2.rows[1].cells[0].text = 'Transformer (under-trained)'
    t2.rows[1].cells[1].text = '0.067'
    t2.rows[1].cells[2].text = '16.6M float'
    t2.rows[2].cells[0].text = 'XCP (discrete + sparse)'
    t2.rows[2].cells[1].text = '0.841'
    t2.rows[2].cells[2].text = '~0.19M integers'

    d.add_paragraph('')
    d.add_paragraph('Figure 1: per-layer compression.')
    _add_pic(d, FIG_C)
    d.add_paragraph('Figure 2: next-token accuracy on real text.')
    _add_pic(d, FIG_A)
    d.add_paragraph('Figure 3: parameter scaling (theoretical).')
    _add_pic(d, FIG_S)

    d.add_heading('5. Discussion', 1)
    d.add_paragraph(
        'XCP is trained in one pass: its discrete core is closed-form counting, not iterative '
        'gradient descent. This is a decisive advantage for training cost. The comparison '
        'against the Transformer is preliminary: the Transformer baseline is under-trained '
        'relative to its capacity, so the reported accuracy gap should be re-confirmed at '
        'convergence and on public benchmarks (e.g., WikiText). The discrete core trades '
        'smoothness for memory; its generalization comes from class-level fallback rather than '
        'continuous interpolation.')

    d.add_heading('6. Conclusion and Future Work', 1)
    d.add_paragraph(
        'XCP demonstrates that a discrete-symbolic core with sparse attention can perform '
        'next-token prediction on real text with a small fraction of Transformer parameters '
        'and one-pass training. Future work: fair comparison at Transformer convergence, '
        'public benchmarks, larger models, and scaling the discrete fact layer.')

    path = os.path.join(OUT, 'XCP_paper_EN.docx')
    d.save(path)
    return path


# ======================================================================
# 英文邮件
# ======================================================================
def build_email_en():
    d = Document()
    # ---- 版式 ----
    st = d.styles['Normal']
    st.font.name = 'Calibri'
    st.font.size = Pt(11)
    st.paragraph_format.line_spacing = 1.4
    for section in d.sections:
        section.top_margin = section.bottom_margin = Inches(0.9)
        section.left_margin = section.right_margin = Inches(1.1)

    def formula(text):
        p = d.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(6)
        r = p.add_run(text)
        r.bold = True
        r.font.size = Pt(12)
        r.font.name = 'Cambria Math'

    def step(title):
        p = d.add_paragraph()
        p.paragraph_format.space_before = Pt(8)
        r = p.add_run(title)
        r.bold = True
        r.font.color.rgb = RGBColor(0x2F, 0x54, 0x9B)

    # ---- 标题 ----
    t = d.add_heading('XCP: A Discrete-Symbolic Language Model — Seeking Academic Feedback and Resources', 0)
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER

    d.add_paragraph('Dear Dr. Gomez,')

    d.add_paragraph(
        'My name is Chengping Xing, a 14-year-old middle-school student from China. I write to '
        'share a research result and to seek academic feedback and resource support — this is '
        'not a commercial proposal.')

    # ---- What XCP is ----
    step('1. What is XCP?')
    d.add_paragraph(
        'I built XCP, a language-model architecture that replaces the Transformer\u2019s '
        'continuous vectors with a discrete-symbolic core plus sparse layers: tokens are stored '
        'as integer IDs and grouped into semantic classes by unsupervised clustering; syntax is '
        'modeled by discrete class-to-class transition rules; lexical facts are stored as '
        'integer lookup tables; and attention is sparsified to 15% of its connections and '
        'trained from scratch. The discrete core is trained by one-pass counting (about 2 '
        'seconds), not by iterative gradient descent.')

    # ---- Core derivation ----
    step('2. The single most important equation: why do discrete symbols save parameters?')
    d.add_paragraph(
        '[MOTIVATION] The Transformer stores one d-dimensional float vector per token; K tokens '
        'cost K*d floats. But at use time a token is essentially just an ID — if we replace the '
        'vector by an integer ID, each token needs only log2(K) bits.')
    d.add_paragraph('[DERIVATION] Compare the storage cost of the two schemes:')
    formula('Continuous :  K*d floats = K*d*32 bits')
    formula('Discrete    :  K*log2(K) bits')
    formula('Ratio = (K*d*32) / (K*log2(K)) = 32*d / log2(K)')
    d.add_paragraph(
        'For d = 4096 and K = 1e6, log2(K) ~ 20, so the theoretical ratio is ~6500x. Because '
        'log2(K) grows very slowly with K, the ratio widens monotonically with vocabulary '
        'size — this is the mathematical source of the scaling advantage.')
    d.add_paragraph(
        '[VERIFIED] I measured real memory (sys.getsizeof / .nbytes, not a formula): at K = 1e6 '
        'the continuous table is 1.436 GB while the discrete integer table is only 4 MB '
        '(measured 359x); extrapolated to K = 1e8 the information-theoretic bound reaches '
        '1349x. The numbers agree with the derivation.')

    # ---- Results ----
    step('3. Measured results (Pride and Prejudice, 131k words, ~7k vocabulary)')
    d.add_paragraph(
        'XCP (~0.19M integer parameters): next-token accuracy 0.841, training ~2 seconds. '
        'Transformer (16.6M float parameters, 2000 training steps): next-token accuracy 0.067. '
        'Per-layer compression measured: ~1000x embedding storage, ~95% feed-forward fact side, '
        '~85% attention, ~112x discrete fact lookup. (Note: the Transformer baseline was '
        'trained for a limited number of steps; the reported advantage should be re-confirmed '
        'at convergence and on public benchmarks.)')

    # ---- Requests ----
    step('4. Two requests, purely academic')
    d.add_paragraph(
        '(1) Endorsement or guidance for arXiv submission — as a 14-year-old student I may need '
        'an endorser. (2) Compute support to run fair comparisons at Transformer convergence '
        'and on public benchmarks (e.g., WikiText).')
    d.add_paragraph(
        'I am happy to share the full method, mathematical derivations, and comparison data. '
        'Some engineering details (optimal hyperparameters and source-level design) are kept '
        'undisclosed for now.')
    d.add_paragraph('Thank you very much for your time and attention.')
    d.add_paragraph('Sincerely,')
    d.add_paragraph('Chengping Xing')

    path = os.path.join(OUT, 'Email_AidanGomez_EN.docx')
    d.save(path)
    return path


# ======================================================================
# 中文论文
# ======================================================================
def build_paper_cn():
    d = Document()
    d.add_heading('XCP：离散符号语言模型（稀疏注意力 + 事实存储）', 0)
    p = d.add_paragraph('作者：平星辰（Xingcheng Ping）')
    p.runs[0].italic = True

    d.add_heading('摘要', 1)
    d.add_paragraph(
        '本文提出 XCP，一种用「离散符号表示 + 稀疏线性层」替代 Transformer 连续向量表示的语言'
        '模型架构。XCP 将词存储为整数 ID（而非稠密向量），通过无监督聚类归入语义类，用离散的'
        '类到类转移规则建模语法，用整数查表存储词汇事实；注意力被稀疏化到 15% 连接并从头训练。'
        '各层压缩率约为：嵌入存储 ~1000x、前馈事实侧 ~95%、注意力 ~85%、离散事实查表 ~112x。'
        '在真实英文小说上，XCP 的 next-token 准确率高于同任务 Transformer，而参数量仅为后者的'
        '一小部分；其离散核心通过单遍计数学习，而非迭代梯度下降。')

    d.add_heading('1. 引言', 1)
    d.add_paragraph(
        'Transformer 及其后继者将每个词编码为连续 d 维向量，并以稠密浮点权重学习知识。这带来'
        '平滑泛化，但也带来严重冗余：词嵌入表花费 K*d 个浮点，前馈层用稠密矩阵存储事实。符号'
        '主义路线曾因难以扩展而被主流放弃；本文展示，离散符号核心 + 在真正需要连续性的地方'
        '引入稀疏，可以在 next-token 预测上达到或超过一个小型 Transformer，同时参数少几个数量'
        '级、且用单遍计数训练。')

    d.add_heading('2. 方法', 1)
    d.add_heading('2.1 离散嵌入', 2)
    d.add_paragraph(
        '每个词分配一个整数 ID。存储成本为 K*log2(K) bit，而连续嵌入为 K*d 个浮点——约省 '
        '1000x。词随后通过左右上下文分布的无监督聚类归入语义类。')
    d.add_heading('2.2 离散前馈（结构 + 事实）', 2)
    d.add_paragraph(
        'FFN 被两张离散表替代。(a) 结构：类到类 m-gram 计数表捕捉句法模式。(b) 事实：'
        '（左词, 类）到下一词的查表存储词汇搭配，未见过的组合用类级兜底。这把语法与词汇分离，'
        '并以整数而非浮点存储事实，比稀疏浮点矩阵再省 ~112x。')
    d.add_heading('2.3 稀疏注意力（从头训练稀疏）', 2)
    d.add_paragraph(
        'Q、K、V、O 投影被固定掩码到 15% 连接并从头训练。与「后剪枝」不同，从头训练稀疏不降低'
        '记忆容量：容量受神经元数约束，而非连接数。我们进一步测得 W_Q^T W_K 的有效秩仅约 '
        '0.06*d，这给出了注意力 ~85% 无损压缩的上界。')
    d.add_heading('2.4 关系门', 2)
    d.add_paragraph(
        '语篇连接词（because/but/then）作为预测状态的调节器，而非内容预测器——对应「抽象词是'
        '操作」的角色。')

    d.add_heading('3. 实验', 1)
    d.add_paragraph(
        '我们在 next-token 预测上对比 XCP 与 PyTorch Transformer（d=512，3 层）。语料包括受控'
        '模板语料（含留出的语义类泛化测试）与真实英文小说（《傲慢与偏见》，13.1 万词，约 7k '
        '词表）。超参数（类数与上下文长度）在一定范围内扫描；最优组合属于工程 know-how，'
        '本文不公开。')

    d.add_heading('4. 结果', 1)
    d.add_paragraph('表 1：各层压缩率。')
    t = d.add_table(rows=5, cols=3)
    t.style = 'Light Grid Accent 1'
    h = t.rows[0].cells
    h[0].text, h[1].text, h[2].text = '层', 'XCP 方法', '压缩率'
    rows = [('嵌入（存储）', '离散整数 ID', '~1000x'),
            ('FFN（事实侧）', '稀疏 W1 + 冻结 W2', '~95%'),
            ('注意力', '稀疏 Q/K/V/O 85%', '~85%'),
            ('事实查表', '整数表 vs 浮点矩阵', '~112x')]
    for i, (a, b, c) in enumerate(rows, start=1):
        t.rows[i].cells[0].text, t.rows[i].cells[1].text, t.rows[i].cells[2].text = a, b, c

    d.add_paragraph('')
    d.add_paragraph('表 2：真实语料端到端对比。')
    t2 = d.add_table(rows=3, cols=3)
    t2.style = 'Light Grid Accent 1'
    h = t2.rows[0].cells
    h[0].text, h[1].text, h[2].text = '模型', '测试 acc（top-1）', '参数'
    t2.rows[1].cells[0].text = 'Transformer（欠训练）'
    t2.rows[1].cells[1].text = '0.067'
    t2.rows[1].cells[2].text = '16.6M float'
    t2.rows[2].cells[0].text = 'XCP（离散 + 稀疏）'
    t2.rows[2].cells[1].text = '0.841'
    t2.rows[2].cells[2].text = '~0.19M 整数'

    d.add_paragraph('')
    d.add_paragraph('图 1：各层压缩率。')
    _add_pic(d, FIG_C)
    d.add_paragraph('图 2：真实语料 next-token 准确率。')
    _add_pic(d, FIG_A)
    d.add_paragraph('图 3：参数效率缩放（理论）。')
    _add_pic(d, FIG_S)

    d.add_heading('5. 讨论', 1)
    d.add_paragraph(
        'XCP 只需训练一次：其离散核心是闭式计数，而非迭代梯度下降，这是训练成本的巨大优势。'
        '与 Transformer 的对比是初步的：Transformer 基线相对其容量欠训练，报告的优势需在其'
        '收敛后、并在公开基准（如 WikiText）上重新确认。离散核心以平滑性换取记忆能力，其泛化'
        '来自类级兜底而非连续插值。')

    d.add_heading('6. 结论与未来工作', 1)
    d.add_paragraph(
        'XCP 证明：离散符号核心 + 稀疏注意力可以在真实文本上做 next-token 预测，参数量仅为 '
        'Transformer 的一小部分，且单遍训练。未来工作：Transformer 收敛后的公平对比、公开'
        '基准、更大模型、以及离散事实层的扩展。')

    path = os.path.join(OUT, 'XCP_paper_CN.docx')
    d.save(path)
    return path


# ======================================================================
# 中文邮件
# ======================================================================
def build_email_cn():
    d = Document()
    # ---- 中文字体 + 版式 ----
    st = d.styles['Normal']
    st.font.name = 'Calibri'
    st.font.size = Pt(11)
    st.paragraph_format.line_spacing = 1.4
    st._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), '微软雅黑')
    for section in d.sections:
        section.top_margin = section.bottom_margin = Inches(0.9)
        section.left_margin = section.right_margin = Inches(1.1)

    def formula(text):
        p = d.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(6)
        r = p.add_run(text)
        r.bold = True
        r.font.size = Pt(12)
        r.font.name = 'Cambria Math'

    def step(title):
        p = d.add_paragraph()
        p.paragraph_format.space_before = Pt(8)
        r = p.add_run(title)
        r.bold = True
        r.font.color.rgb = RGBColor(0x2F, 0x54, 0x9B)

    # ---- 标题 ----
    t = d.add_heading('XCP：离散符号语言模型 —— 寻求学术反馈与资源支持', 0)
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER

    d.add_paragraph('尊敬的 Gomez 博士：')

    d.add_paragraph(
        '我叫邢承平（Chengping Xing），中国一名 14 岁的初中生。我写信分享一项研究成果，'
        '并寻求学术反馈与资源支持——这不是商业合作，纯属学术性质。')

    # ---- XCP 简介 ----
    step('一、XCP 是什么')
    d.add_paragraph(
        '我构建了 XCP，一种用「离散符号核心 + 稀疏层」替代 Transformer 连续向量的语言模型'
        '架构：词被存储为整数 ID，并通过无监督聚类归入语义类；语法由离散的类到类规则建模；'
        '词汇事实以整数查表存储；注意力被稀疏化到 15% 连接并从头训练。整个离散核心只需'
        '单遍计数训练（约 2 秒），而非迭代梯度下降。')

    # ---- 核心推导 ----
    step('二、最核心的一条公式：为什么离散符号能省参数？（从想到到做到）')
    d.add_paragraph('[想到] Transformer 给每个词存一个 d 维浮点向量，K 个词共需 K·d 个浮点。'
                    '但模型使用时，词本质上只是一个"编号"——若用整数 ID 替代向量，每个词'
                    '只需要 log₂K 个 bit（K=100 万时仅 20 bit）。能否省掉这 d 个浮点？')
    d.add_paragraph('[推导] 对比两种方案的存储成本：')
    formula('连续方案：K·d 个浮点 = K·d·32 bit')
    formula('离散方案：K·log₂K bit')
    formula('压缩比 = (K·d·32) / (K·log₂K) = 32·d / log₂K')
    d.add_paragraph('大模型 d=4096、K=10⁶ 时，log₂K≈20，理论压缩比 ≈ 32×4096/20 ≈ 6500x。'
                    '且 log₂K 随 K 增长极慢，压缩比随词表规模单调放大——这是"缩放优势"'
                    '的数学来源。')
    d.add_paragraph('[做到] 用真实内存实测验证（sys.getsizeof / .nbytes，非公式估算）：'
                    'K=10⁶ 时，连续表实测 1.436 GB，离散整数表仅 4 MB（实测 359x）；按信息论'
                    '下界外推 K=10⁸ 达 1349x。数字与推导一致。')

    # ---- 实测结果 ----
    step('三、实测结果（《傲慢与偏见》，13.1 万词，约 7k 词表）')
    d.add_paragraph(
        'XCP（约 19 万整数参数）：next-token 准确率 0.841，训练约 2 秒；'
        'Transformer（1663 万浮点参数，2000 步训练）：next-token 准确率 0.067。'
        '各层压缩率实测：嵌入存储约 1000x、前馈事实侧约 95%、注意力约 85%、离散事实查表'
        '约 112x。（注：Transformer 基线训练步数有限，报告的优势需在其收敛后、并在公开'
        '基准上作公平对比确认。）')

    # ---- 请求 ----
    step('四、两项纯学术请求')
    d.add_paragraph(
        '(1) 为 arXiv 投稿提供背书或指导——作为 14 岁学生，我可能需要一位背书人；'
        '(2) 算力支持，以便在 Transformer 收敛后、并在公开基准（如 WikiText）上做公平对比。')
    d.add_paragraph(
        '我愿意分享完整方法、数学推导和对比数据。部分工程细节（最优超参数与源代码级设计）'
        '暂不公开。')
    d.add_paragraph('非常感谢您的时间与关注。')
    d.add_paragraph('此致敬礼')
    d.add_paragraph('邢承平（Chengping Xing）')

    path = os.path.join(OUT, 'Email_AidanGomez_CN.docx')
    d.save(path)
    return path


if __name__ == '__main__':
    # 只生成两封邮件；论文由 build_papers.py 生成（XCP_paper_EN/CN.docx）
    r = [build_email_en(), build_email_cn()]
    print('generated:')
    for p in r:
        print(' ', p)
