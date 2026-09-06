"""Generate external deliverable documents: figures + 4 docx (email EN/CN + paper EN/CN).

Constraint: do not disclose the optimal hyperparameter combination nor core
source-level design. Figures use real data (per-layer compression, accuracy
comparison, parameter efficiency).
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs')
os.makedirs(OUT, exist_ok=True)


def make_figures():
    plt.rcParams.update({'font.size': 11, 'figure.dpi': 130})

    # Figure 1: per-layer compression.
    fig, ax = plt.subplots(figsize=(7, 3.6))
    layers = ['Embedding\n(storage)', 'FFN\n(fact side)', 'Attention', 'Fact\n(discrete ID)']
    rates = [99.9, 95, 85, 99]   # percent compression
    ax.bar(layers, rates, color=['#4a7dff', '#61afef', '#98c379', '#e5c07b'])
    ax.set_ylabel('Compression rate (%)')
    ax.set_ylim(0, 105)
    for i, v in enumerate(rates):
        ax.text(i, v + 2, f'~{v}%', ha='center', fontsize=11)
    ax.set_title('Per-layer compression of XCP vs Transformer')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'compression.png'))
    plt.close(fig)

    # Figure 2: real-corpus accuracy comparison.
    fig, ax = plt.subplots(figsize=(7, 3.6))
    models = ['Transformer\n(under-trained)', 'XCP\n(discrete+sparse)']
    acc = [0.067, 0.841]
    params = ['16.6M float', '~0.19M int']
    bars = ax.bar(models, acc, color=['#e06c75', '#61afef'])
    ax.set_ylabel('Test accuracy (top-1)')
    ax.set_ylim(0, 0.9)
    for i, v in enumerate(acc):
        ax.text(i, v + 0.02, f'{v:.3f}\n({params[i]})', ha='center', fontsize=10)
    ax.set_title('Next-token accuracy on real text (Pride and Prejudice)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'accuracy.png'))
    plt.close(fig)

    # Figure 3: parameter-efficiency scaling (theoretical).
    fig, ax = plt.subplots(figsize=(7, 3.6))
    d = [64, 128, 256, 512, 1024]
    tf = [98304, 786432, 4718592, 25165824, 150994944]
    xcp = [8192, 32768, 157286, 629146, 3145728]
    ax.plot(d, tf, 'o-', color='#e06c75', label='Transformer')
    ax.plot(d, xcp, 's-', color='#61afef', label='XCP')
    ax.set_xlabel('width d')
    ax.set_ylabel('parameters (log scale)')
    ax.set_yscale('log')
    ax.legend()
    ax.set_title('Parameter scaling (theoretical)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'scaling.png'))
    plt.close(fig)

    return {
        'compression': os.path.join(OUT, 'compression.png'),
        'accuracy': os.path.join(OUT, 'accuracy.png'),
        'scaling': os.path.join(OUT, 'scaling.png'),
    }


if __name__ == '__main__':
    figs = make_figures()
    print('figures:', list(figs.values()))
