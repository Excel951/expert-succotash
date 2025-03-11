import numpy as np
import pandas as pd

logit_awal=-0.679

# Buat DataFrame dengan perhitungan lengkap untuk disimpan ke dalam Excel
data_perhitungan = {
    "Sample": [1, 2, 3],
    "Salary": [62506, 104437, 64955],
    "EngagementSurvey": [4.6, 4.96, 3.02],
    "Absences": [1, 17, 3],
    "Termd (Target)": [0, 1, 1],
    "Logit Awal": [logit_awal] * 3,
    "Probabilitas Awal": [1 / (1 + np.exp(-logit_awal))] * 3,
}

# Hitung gradien dan hessian
data_perhitungan["Gradien"] = [p - y for p, y in zip(data_perhitungan["Probabilitas Awal"], data_perhitungan["Termd (Target)"])]
data_perhitungan["Hessian"] = [p * (1 - p) for p in data_perhitungan["Probabilitas Awal"]]

# Simulasikan bobot pembaruan (misalnya berdasarkan leaf node)
data_perhitungan["Bobot Pembaruan"] = [-0.1, 0.2, 0.15]  # Contoh bobot
data_perhitungan["Logit Baru"] = [logit + w for logit, w in zip(data_perhitungan["Logit Awal"], data_perhitungan["Bobot Pembaruan"])]

# Buat DataFrame
df_perhitungan = pd.DataFrame(data_perhitungan)

# Simpan ke Excel
excel_path = "perhitungan_xgboost.xls"
df_perhitungan.to_excel(excel_path)
# with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
#     df_perhitungan.to_excel(writer, sheet_name="XGBoost Calculation", index=False)

# Berikan path file untuk diunduh
# excel_path