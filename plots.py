import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_double_descent(results, n_train, lam, save_path='double_descent.pdf'):
    d = np.array(results['d'])
    ls_train  = np.array(results['ls_train'])
    ls_test   = np.array(results['ls_test'])
    r_train   = np.array(results['ridge_train'])
    r_test    = np.array(results['ridge_test'])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Double Descent — n_train={n_train}', fontsize=14)

    # Normalize by d/n (model complexity ratio)
    ratio = d / n_train

    for ax, x, xlabel in zip(
        axes,
        [d, ratio],
        ['Model dimension $d$', 'Complexity ratio $d/n$']
    ):
        ax.plot(x, ls_train, '--', color='steelblue',  label='LS train',   linewidth=1.5)
        ax.plot(x, ls_test,  '-',  color='steelblue',  label='LS test',    linewidth=2)
        ax.plot(x, r_train,  '--', color='tomato',     label='Ridge train', linewidth=1.5)
        ax.plot(x, r_test,   '-',  color='tomato',     label='Ridge test',  linewidth=2)

        # Mark interpolation threshold
        thresh_x = n_train / n_train if xlabel.startswith('Complex') else n_train
        ax.axvline(thresh_x, color='gray', linestyle=':', linewidth=1.5,
                   label=f'Interp. threshold ($d=n$)')

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('MSE', fontsize=12)
        ax.legend(fontsize=10)
        ax.set_ylim(bottom=0, top=min(10, np.percentile(ls_test, 98) * 2))
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()


def plot_noise_comparison(all_results, n_train, save_path='noise_comparison.pdf'):
    noise_levels = list(all_results.keys())
    colors = plt.colormaps['viridis'](np.linspace(0.1, 0.9, len(noise_levels)))

    fig, ax = plt.subplots(figsize=(9, 5))
    for sigma, color in zip(noise_levels, colors):
        res = all_results[sigma]
        d = np.array(res['d'])
        ls_test = np.array(res['ls_test'])
        ax.plot(d / n_train, ls_test, color=color, label=f'σ={sigma}', linewidth=2)

    ax.axvline(1.0, color='gray', linestyle=':', linewidth=1.5, label='Threshold $d=n$')
    ax.set_xlabel('Complexity ratio $d/n$', fontsize=12)
    ax.set_ylabel('Test MSE (Least Squares)', fontsize=12)
    ax.set_title('Effect of Noise on Double Descent', fontsize=13)
    ax.set_ylim(bottom=0, top=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()