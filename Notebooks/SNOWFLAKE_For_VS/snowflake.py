from __future__ import print_function, division

import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns


from scipy.stats import probplot, norm, zscore, binom_test
from scipy.special import logit, expit


def snowflake(scores, do_plot=False):
    """Snowflake Analysis. Assume scores are bounded 0-1 scores with higher
    values being more favorable"""
    fitmean, fitstd = norm.fit(logit(scores))
    zs = zscore(logit(scores))
    x = np.linspace(-5, 5, 1000)

    osm, osr = probplot(zs, fit=False, plot=None)
    diff = osr-osm
    stop_pts = np.where(diff < 0.0)[0]
    snowflake_z = osm[stop_pts[-1]]

    obs_count = np.asarray([np.sum(zs >= i) for i in x])
    binom_std = np.sqrt(len(zs) * norm.sf(x)*(1-norm.sf(x)))

    diff = obs_count - (norm.sf(x)*len(zs) + 2*binom_std)
    x_idx = np.argwhere(x < snowflake_z)[-1][0]
    stop_pts = np.where(diff[x_idx:] > 0.0)[0]
    snowflake95_z = x[x_idx+stop_pts[0]]

    snowflake_cutoff = expit(snowflake_z*fitstd+fitmean)
    prob = norm.sf(snowflake_z)
    print("SNOWFLAKE cutoff: {:.3f}".format(snowflake_cutoff))
    print("Probability of an inactive scoring above SNOWFLAKE: {:.4f}".format(
        prob))

    n_above_cutoff = np.sum(zs >= snowflake_z)
    print("{} scores above the SNOWFLAKE cutoff out of {}".format(
        n_above_cutoff, len(zs)))

    print("p-value of One-sided Binomial Test: {:.4f}".format(
        binom_test(n_above_cutoff, len(zs), prob, alternative="greater")))

    print("\n")
    snowflake95_cutoff = expit(snowflake95_z*fitstd+fitmean)
    n_above_cutoff = np.sum(zs >= snowflake95_z)
    prob = norm.sf(snowflake95_z)
    print("SNOWFLAKE95 cutoff: {:.3f}".format(snowflake95_cutoff))
    print("Probability of an inactive scoring above SNOWFLAKE95: {:.4f}".format(
        prob))

    print("{} scores above the SNOWFLAKE95 cutoff out of {}".format(
        n_above_cutoff, len(zs)))
    print("p-value of One-sided Binomial Test: {:.4f}".format(
        binom_test(n_above_cutoff, len(zs), prob, alternative="greater")))
    print("")

    if do_plot:
        plt.hist(scores , bins=50, histtype='stepfilled', color='k', alpha=0.4, label="Scores")
        plt.vlines(snowflake_cutoff, 0, len(scores)*0.05, linestyles="--", linewidth=4, label="SNOWFLAKE: {:.3f}".format(snowflake_cutoff))
        plt.vlines(snowflake95_cutoff, 0, len(scores)*0.05, linestyles=":", linewidth=4, label="SNOWFLAKE95: {:.3f}".format(snowflake95_cutoff))
        plt.title("Score Distribution")
        plt.legend(loc="upper right")
        plt.show()
        plt.close()

        plt.plot(x, x, 'k--', alpha=0.3)
        plt.plot(osm, osr, 'b.', alpha=0.4)
        plt.vlines(snowflake_z, 0, snowflake_z*2, linestyles="--", linewidth=4, label="SNOWFLAKE".format(snowflake_z))
        plt.vlines(snowflake95_z, 0, snowflake95_z*2, linestyles=":", linewidth=4, label="SNOWFLAKE95".format(snowflake95_z))
        plt.title("QQ Plot")
        plt.legend(loc="upper left")
        plt.show()
        plt.close()

    return (snowflake_cutoff, snowflake95_cutoff)
