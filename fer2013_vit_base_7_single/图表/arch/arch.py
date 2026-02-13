import os
from graphviz import Digraph
from PIL import Image, ImageEnhance, ImageOps, ImageDraw

# 强制指定路径
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'


def generate_compact_ultimate_architecture(img_path):
    print(">>> [日志] 正在执行“宽屏紧凑化”布局优化...")
    # rankdir='LR' 重新回归横向，但通过控制 nodesep 让它变矮
    dot = Digraph('FER_Compact_Ultimate', format='png')
    dot.attr(rankdir='LR', nodesep='0.2', ranksep='0.4', fontname='Times New Roman')

    temp_dir = 'ultimate_temp'
    os.makedirs(temp_dir, exist_ok=True)

    # --- 图像处理（完全保留您的逻辑） ---
    orig = Image.open(img_path).convert('RGB').resize((64, 64))
    orig.save(os.path.join(temp_dir, 'p0.png'))
    t1 = orig.rotate(25).transpose(Image.FLIP_LEFT_RIGHT)
    t1.save(os.path.join(temp_dir, 'p1.png'))
    t2 = ImageOps.solarize(orig, threshold=100)
    t2.save(os.path.join(temp_dir, 'p2.png'))
    t3 = orig.copy()
    ImageDraw.Draw(t3).rectangle([10, 10, 35, 35], fill=(0, 0, 0))
    t3.save(os.path.join(temp_dir, 'p3.png'))

    # --- Step 1: MDT (左侧) ---
    with dot.subgraph(name='cluster_step1') as c:
        c.attr(label='Step 1: MDT Strategy', color='orange', style='dashed', fontname='Times-Bold')
        c.node('input',
               label=f'<<TABLE BORDER="0"><TR><TD><IMG SRC="{os.path.join(temp_dir, "p0.png")}"/></TD></TR><TR><TD>Input</TD></TR></TABLE>>',
               shape='none')

        # 保持三个分支垂直排列，以减小横向宽度
        c.node('b1',
               label=f'<<TABLE BORDER="0"><TR><TD><IMG SRC="{os.path.join(temp_dir, "p1.png")}"/></TD></TR><TR><TD>Geo.</TD></TR></TABLE>>',
               shape='none')
        c.node('b2',
               label=f'<<TABLE BORDER="0"><TR><TD><IMG SRC="{os.path.join(temp_dir, "p2.png")}"/></TD></TR><TR><TD>Photo.</TD></TR></TABLE>>',
               shape='none')
        c.node('b3',
               label=f'<<TABLE BORDER="0"><TR><TD><IMG SRC="{os.path.join(temp_dir, "p3.png")}"/></TD></TR><TR><TD>Occ.</TD></TR></TABLE>>',
               shape='none')

        c.node('agg', 'Aggregator', shape='cylinder', style='filled', fillcolor='#FFE0B2', height='0.3')
        c.node('sampler', 'Sampler', shape='note', fillcolor='#FFF9C4', fontsize='10')

        c.edge('input', 'b1');
        c.edge('input', 'b2');
        c.edge('input', 'b3')
        c.edge('b1', 'agg');
        c.edge('b2', 'agg');
        c.edge('b3', 'agg')
        c.edge('agg', 'sampler')

    # --- Step 2: Backbone (中间) ---
    with dot.subgraph(name='cluster_step2') as v:
        v.attr(label='Step 2: ViT-Base', color='blue', fontname='Times-Bold')
        v.node('patch', 'Patch\nEmbed', shape='rect', style='filled', fillcolor='#D0E8FF')
        v.node('vit', 'Transformer Encoder\n(L=12, MSA+MLP)', shape='box3d', style='filled', fillcolor='#64B5F6',
               fontcolor='white')
        v.node('llrd', 'LLRD', shape='parallelogram', fillcolor='#E3F2FD', fontsize='9')
        v.edge('patch', 'vit')
        v.edge('llrd', 'vit', style='dotted', dir='back')

    # --- Step 3: Head (右侧) ---
    with dot.subgraph(name='cluster_step3') as s:
        s.attr(label='Step 3: SSH & Fusion', color='darkgreen', fontname='Times-Bold')
        # 垂直堆叠 SSH 和 CLS 以节省横向空间
        s.node('ssh', 'SSH Module\n(AU Topology)', shape='hexagon', style='filled', fillcolor='#FFCC80', penwidth='1.5')
        s.node('cls', 'CLS Token', shape='circle', style='filled', fillcolor='#A5D6A7')
        s.node('fusion', 'Fusion', shape='doublecircle', fillcolor='#F5F5F5')
        s.node('out', 'Prediction\n[74.21%]', shape='trapezium', style='filled', fillcolor='#EF9A9A')

        s.edge('ssh', 'fusion');
        s.edge('cls', 'fusion')
        s.edge('fusion', 'out')

    # --- 跨模块连接 (关键：通过增加约束优化布局) ---

    # 增加 penwidth 和颜色，让主干流更明显
    dot.edge('sampler', 'patch', label=' x_aug', penwidth='2.0', color='orange')

    # 从 Vit 分叉到 SSH 和 CLS
    dot.edge('vit', 'ssh', color='blue')
    dot.edge('vit', 'cls', color='blue')

    # 渲染
    dot.render('FER_Compact_Ultimate_Layout', cleanup=True)
    print(">>> [成功] 宽屏紧凑版已生成：FER_Compact_Ultimate_Layout.png")


if __name__ == '__main__':
    path = r'C:/Users/JSPG/code/github/facial-affect-transformer/images/images/happy/158.jpg'
    generate_compact_ultimate_architecture(path)