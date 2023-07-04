import pandas as pd
import matplotlib.pyplot as plt

def results_to_files(results_dict, output_dir):
    results_df = pd.DataFrame(results_dict).set_index('epoch')

    for measurement in results_df.columns:
        results_df[measurement].plot(title=measurement.replace('_', ' '))
        plt.savefig(f'{output_dir}/{measurement}.jpg')
        plt.cla()

    results_df.to_csv(f'{output_dir}/results.csv', index=True)