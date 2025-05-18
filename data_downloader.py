import pandas as pd

# ------------------ Washington Dataset -----------------

# Download
csv_url = "https://ocio.wa.gov/sites/default/files/Data_Breach_Notifications.csv"
wa_df = pd.read_csv(csv_url)

# Saving
wa_df.to_csv("data/washington.csv", index=False)

# ------------------ Privacy Rights Dataset -----------------

# Download
prc_url = "https://privacyrights.org/sites/default/files/chronology.csv"
prc_df = pd.read_csv(prc_url)

# Saving
prc_df.to_csv("data/privacy_rights_clearinghouse.csv", index=False)


# ------------------ Verizon Dataset -----------------

# Downloading
verizon_url = "https://www.verizon.com/business/resources/reports/dbir/2024/2024-DBIR.pdf"
r = requests.get(verizon_url)

# Saving
with open("verizon_dbir_2024.pdf", "wb") as f:
    f.write(r.content)