import pandas as pd

from run import OUTPUT_DIR, PREFIX

importance = pd.concat(
    [
        pd.read_csv(f"{OUTPUT_DIR}/{PREFIX}_{year}_{year+1}.csv")
        for year in range(17, 24)
    ]
)
importance.to_csv(f"{OUTPUT_DIR}/{PREFIX}_17_24.csv", index=False)
