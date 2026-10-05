import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


metrics = ['WER', 'WIL', 'SemDist']
dataset_files = {
    "SAA": "CSV/df_files/df_SAA.csv",
    "L2Arc": "CSV/df_files/df_L2Arc.csv",
    "Fair-Speech": "CSV/df_files/df_FairSpeech.csv",
    "EdAcc": "CSV/df_files/df_EdAcc.csv",
    "ALLSSTAR": "CSV/df_files/df_ALLSSTAR.csv",
    "Afri200": "CSV/df_files/df_Afrispeech.csv"
}
models = ['Whisper-Small', 'Whisper-Large', 'ParakeetV3', 'CanaryV2' ]

results = []
for model_name in models:
    def load_and_process(name, path):
        df = pd.read_csv(path)
        df['Dataset'] = name
        df = df[df['Model'] == model_name]
        df["WER"] = MinMaxScaler().fit_transform(df[["WER"]])
        df["WIL"] = MinMaxScaler().fit_transform(df[["WIL"]])
        df["SemDist"] = MinMaxScaler().fit_transform(df[["SemDist"]])
        df['LanguageDist'] = MinMaxScaler().fit_transform(df[["LanguageDist"]]) 
        return df

    # ==============================
    # Load datasets
    # ==============================
    datasets = {name: load_and_process(name, path)
                for name, path in dataset_files.items()}

    for metric in metrics:
        all_data = []
        for dataset_name, df in datasets.items():
            df = df.dropna(subset=["LanguageDist", metric])
            Groudp_df = df.groupby("LanguageDist")[f"{metric}"].mean().reset_index()
            all_data.append(df)
            
        df_model = pd.concat(
                    all_data,
                    ignore_index=True
                )
        results.append(df_model)
results_df = pd.concat(
    results,
    ignore_index=True
)
import gpboost as gpb
import statsmodels.api as sm
from scipy.stats import norm
import gpboost as gpb
import statsmodels.api as sm
from scipy.stats import norm
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

for metric in metrics:
    print(f"\n===== {metric} =====")
    grouped = (
        results_df
        .groupby(["L1", "Dataset", "Model"], as_index=False)
        .agg(
            LanguageDist=("LanguageDist", "first"),
            **{
            f"{metric}_mean": (metric, "mean"),
            f"{metric}_std": (metric, "std"),
            f"{metric}_var": (metric, "var"),
            },
            Language_Family=("Language_Family", "first"),
            sample_count=("L1", "count")
        )
    )

    dataset_colors = {
        "SAA": "#7d9db4",
        "L2Arc": "#759375",
        "Fair-Speech": "#bc7272",
        "EdAcc": "#c9a472",
        "ALLSSTAR": "#7b7b7b",
        "Afri200": "#b4a7d6"
    }

    models_list = results_df["Model"].unique()
    n_models = len(models_list)

    # Grid layout: adjust nrows/ncols to fit however many models you have
    ncols = 2
    nrows = int(np.ceil(n_models / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharey=True)
    axes = np.array(axes).reshape(-1)  # flatten in case of 1D/2D mismatch
    
    for idx, model in enumerate(models_list):
        ax = axes[idx]
        df_model = results_df[results_df["Model"] == model]
        
        print(f"\n{model}")
    
        X = sm.add_constant(df_model["LanguageDist"])
        gp_model = gpb.GPModel(group_data=df_model["Dataset"], likelihood="tweedie")
        gp_model.fit(y=df_model[metric], X=X)
        print(gp_model.summary())
        coef_table = gp_model.get_coef(std_err=True)
        beta = coef_table.loc[coef_table.index[0], "LanguageDist"]
        se = coef_table.loc[coef_table.index[1], "LanguageDist"]
        z = beta / se
        pval = 2 * (1 - norm.cdf(abs(z)))
    
        x_grid = np.linspace(df_model["LanguageDist"].min(), df_model["LanguageDist"].max(), 100)
    
        for dataset in df_model["Dataset"].unique():
            X_grid = sm.add_constant(
                pd.DataFrame({"LanguageDist": x_grid}),
                has_constant="add"
            )
            group_grid = np.array([dataset] * len(x_grid))
    
            pred = gp_model.predict(X_pred=X_grid, group_data_pred=group_grid)
            ax.plot(x_grid, pred["mu"], label=dataset, color=dataset_colors[dataset], alpha=0.8, linewidth=2)
    
            subset = grouped[(grouped["Dataset"] == dataset) & (grouped["Model"] == model)]
            ax.scatter(
                subset["LanguageDist"],
                subset[f"{metric}_mean"],
                color=dataset_colors[dataset],
                s=12,
                alpha=0.7
            )
    
        ax.set_title(model, fontsize=15, fontweight="bold")
    
        sig_stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        annotation = (
            f"β = {beta:.3f} (SE = {se:.3f})\n"
            f"p = {pval:.4g} {sig_stars}"
        )
        ax.text(
            0.02, 0.98, annotation,
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray")
        )
        sns.despine(ax=ax)
    
    # Hide any unused subplot axes (if n_models doesn't fill the grid evenly)
    for j in range(n_models, len(axes)):
        axes[j].set_visible(False)
        axes[j].legend(False)
    
    # One shared x and y label for the whole figure instead of per-subplot
    fig.supxlabel(r"Linguistic Distance (LD)", fontsize=18)
    fig.supylabel(metric, fontsize=18)
    
    # One shared legend for the whole figure instead of per-panel
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Dataset", bbox_to_anchor=(0.85, 0.4), loc="center left",
            title_fontsize=12, fontsize=11)

    plt.tight_layout()
    plt.savefig(f"Plot/GLMMs_{metric}_LanguageDist_all_models.png", dpi=300, bbox_inches="tight")
    plt.show()