import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

metrics = ['WER', 'WIL', 'SemDist']

dataset_files = {
    "SAA": "CSV/df_files/df_SAA.csv",
    "L2Arc": "CSV/df_files/df_L2Arc.csv",
    "FairSpeech": "CSV/df_files/df_FairSpeech.csv",
    "EdAcc": "CSV/df_files/df_EdAcc.csv",
    "ALLSSTAR": "CSV/df_files/df_ALLSSTAR.csv",
    "Afri200": "CSV/df_files/df_Afrispeech.csv"
}
models = ['Whisper-Small', 'Whisper-Large', 'ParakeetV3', 'CanaryV2' ]
results = []
for model_name in models:
    print(f"\n===== {model_name} =====")
    def load_and_process(path):
        df = pd.read_csv(path, index_col=0)
        df = df[df['Model'] == model_name]
        # Normalize metrics and linguistic distance
        for metric in ["WER", "WIL", "SemDist"]:
            df[metric] = MinMaxScaler().fit_transform(df[[metric]])
        df["LanguageDist"] = MinMaxScaler().fit_transform(
            df[["LanguageDist"]]
        ) 
        return df

    # ==============================
    # Load datasets
    # ==============================
    datasets = {name: load_and_process(path)
                for name, path in dataset_files.items()}

    for metric in metrics:
        print(f"---- {metric} ----")
        for dataset_name, df in datasets.items():
            df_model = df.dropna(subset=["LanguageDist", metric])
            df_group = (
                df_model.groupby("L1")
                .agg({
                    "LanguageDist": "first",
                    metric: "mean",
                    "L1": "count"
                })
                .rename(columns={"L1": "sample_count"})
                .reset_index()
            )
            X = sm.add_constant(df_group["LanguageDist"])
            y = df_group[metric]

            model = sm.OLS(y, X).fit()
            
            beta = model.params["LanguageDist"]
            pval = model.pvalues["LanguageDist"]
            r2 = model.rsquared

            print(f"{dataset_name}: β={beta:.4f}, p={pval:.4g}, R²={r2:.3f}")
            beta = model.params["LanguageDist"]
            ci_low, ci_high = model.conf_int().loc["LanguageDist"]


            results.append({
                "Metric": metric,
                "Dataset": dataset_name,
                "Model": model_name,
                "Beta": round(beta,3),
                "CI_low": ci_low,
                "CI_high": ci_high,
                "r2": round(r2,3),
                "pval": round(pval,5)
            })

results_df = pd.DataFrame(results)

def get_significance_marker(row):
    """
    Returns a significance marker string based on p-value if available,
    otherwise falls back to whether the CI excludes zero.
    """
    if 'pval' in row.index and not np.isnan(row['pval']):
        p = row['pval']
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return ''
    else:
        # Fallback: significant if CI doesn't include 0
        if row['CI_low'] > 0 or row['CI_high'] < 0:
            return '*'
        return ''
def grouped_bar_plot(df, metrics):
    """
    Grouped bar plot where the legend shows only model (marker/hatch) without color.
    """
    # Colors per dataset
    dataset_colors = {
        "SAA": "#7d9db4",
        "L2Arc": "#759375",
        "FairSpeech": "#bc7272",
        "EdAcc": "#c9a472",
        "ALLSSTAR": "#7b7b7b",
        "Afri200": "#b4a7d6"
    }

    # Hatch per model
    model_hatches = {
        "Whisper-Small": "",
        "Whisper-Large": "//",
        "ParakeetV3": "oo",
        "CanaryV2": "++"
    }

    models = df['Model'].unique()
    datasets = df['Dataset'].unique()
    
    n_metrics = len(metrics)

    fig, axes = plt.subplots(n_metrics, 1, figsize=(14, 4*n_metrics), sharex=True)
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        sub = df[df['Metric'] == metric]
        x = np.arange(len(datasets))
        width = 0.2

        for i, model in enumerate(models):
            vals = []
            errs = []
            r2_vals = []
            sig_marks = []
            for dataset in datasets:
                row = sub[(sub['Model']==model) & (sub['Dataset']==dataset)]
                if not row.empty:
                    vals.append(row['Beta'].values[0])
                    r2_vals.append(row['r2'].values[0])
                    errs.append([[row['Beta'].values[0]-row['CI_low'].values[0]],
                                 [row['CI_high'].values[0]-row['Beta'].values[0]]])
                    sig_marks.append(get_significance_marker(row.iloc[0]))
                else:
                    vals.append(0)
                    errs.append([[0],[0]])
            errs = np.array(errs).squeeze().T

            bars = ax.bar(x + i*width, vals, width=width, 
                   color=[dataset_colors[d] for d in datasets],
                   yerr=errs, capsize=4,
                   hatch=model_hatches[model],
                   alpha=0.75,
                   edgecolor='black')
            for idx, (val, mark) in enumerate(zip(vals, sig_marks)):
                if not mark:
                    continue
                bar_x = x[idx] + i * width
                if val >= 0:
                    star_y = val + errs[1, idx] + 0.003  # a bit above upper CI
                    va_dir = 'bottom'
                else:
                    star_y = val - errs[0, idx] - 0.003  # a bit below lower CI
                    va_dir = 'top'

                ax.text(
                    x=bar_x,
                    y=star_y,
                    s=mark,
                    ha='center',
                    va=va_dir,
                    fontsize=22,
                    fontweight='bold',
                    color='black'
                )

            if bars.errorbar:
                # bars.errorbar[2] contains the line collections (caps and stems)
                for line_collection in bars.errorbar[2]:
                    line_collection.set_alpha(0.4)
            for idx, (val, r2) in enumerate(zip(vals, r2_vals)):
                if np.isnan(r2):  # Skip labeling if no data exists
                    continue
                
                # Compute the precise centered X coordinate for this specific bar
                bar_x = x[idx] + i * width
                stagger_padding = 0.08 if (i % 2 == 0) else 0.007
                bar_x += -0.05 if (i / 2 < 1) else 0.05
                stat = 0.05
                
                if metric == 'WIL':
                    stat *= 1.3
                    stagger_padding*= 1.3
                if metric == 'WER':
                    stat *= 0.8
                    stagger_padding*= 0.9
                if metric == 'SemDist':
                    stat *= 0.9
                
                if val >= 0:
                    text_y = - stat - stagger_padding 
                    va_dir = 'top'
                else:
                    text_y = 0.02 + stagger_padding 
                    va_dir = 'bottom'
                
                
                ax.text(
                    x=bar_x, 
                    y=text_y , 
                    s=f"{r2:.2f}",
                    ha='center', 
                    va=va_dir,
                    fontsize=24, 
                    color=dataset_colors[datasets[idx]]
                )


        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.tick_params(axis='y', labelsize=25)
        ax.set_xticks(x + width)  # center x-ticks
        ax.set_yticks(np.arange(-0.2, 0.21, 0.2))
        ax.set_xticklabels(datasets, fontsize=25)
        ax.set_ylabel(r"$\beta$ (Linguistic Distance)", fontsize=30)
        if metric!='WIL':
            ax.set_ylabel("")
        
        ax.set_title(metric, fontsize=30)

    # Custom legend for models only (black color, just hatch)
    legend_handles = [plt.Rectangle((0,0),1,1, facecolor='white', edgecolor='black', hatch=hatch, label=model)
                      for model, hatch in model_hatches.items()]
    
    axes[2].legend(handles=legend_handles, title="Model",ncol=4, bbox_to_anchor=(-0.07, -0.2),columnspacing=0.8, handletextpad=0.4, title_fontsize=24,fontsize=23,loc='upper left')

    plt.tight_layout()
    plt.savefig("Plot/beta.pdf")
    plt.show()

grouped_bar_plot(results_df, metrics)
