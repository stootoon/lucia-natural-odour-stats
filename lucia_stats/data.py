import pandas as pd
from scipy.stats import zscore

spreadsheet_id = "1rF6KRdGkyQq3VtTHx7fhZoprIboU4AooWy5Jqx5fbeg"
SHEET_URL = lambda gid: f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv&gid={gid}"

def get_noni():
    noni_data_gid = 365555364
    noni_url = SHEET_URL(noni_data_gid)
    df = pd.read_csv(noni_url)
    df = df.set_index('Odour')
    column_tuples = [col.split('_rep') for col in df.columns]
    df.columns = pd.MultiIndex.from_tuples(column_tuples, names=['Sample', 'Replicate'])
    return df

def get_noni_ripeness(do_zscore=False):
    df = get_noni()

    dsec_avgs_gid = "522303434"
    dsec = pd.read_csv(SHEET_URL(dsec_avgs_gid), header=None)
    mask = dsec.iloc[1].notna() & (dsec.columns != 0) 
    cols = dsec.columns[mask]
    sample_ids = dsec.iloc[1][cols].str.replace(r'^Avg\s+', '', regex=True).values
    ripeness = dsec.iloc[0][cols].values
    # Create a dictionary mapping from sample ID to ripeness
    sample_ripeness = dict(zip(sample_ids, ripeness))

    tidy = df.T.reset_index()
    tidy.columns.name = None
    tidy["Ripeness"] = tidy["Sample"].map(sample_ripeness)
    # add a "Ripeness" column which inserts the value based using the  
    missing = set(tidy['Sample']) - set(sample_ripeness) 
    assert not missing, f"No ripeness for: {missing}" 
    tidy.insert(2, 'Ripeness', tidy.pop('Ripeness'))
    tidy["Replicate"] = tidy["Replicate"].astype(int)
    tidy["Ripeness"] = tidy["Ripeness"].astype(int)

    meta_cols = ["Sample", "Replicate", "Ripeness"]
    # Strip trailing or leading whitespace from column names
    tidy.columns = tidy.columns.str.strip()
    odour_cols = tidy.columns.difference(meta_cols)

    if do_zscore:
       tidy[odour_cols] = zscore(tidy[odour_cols].values, axis=1, nan_policy='omit')

    return tidy, odour_cols


def get_pandan():
    pandan_data_gid = 1369326582
    pandan_url = SHEET_URL(pandan_data_gid)
    df = pd.read_csv(pandan_url)
    df = df.set_index('Odour')
    column_tuples = [col.split('_rep') for col in df.columns]
    df.columns = pd.MultiIndex.from_tuples(column_tuples, names=['Sample', 'Replicate'])
    return df

def tidy_pandan(do_zscore=False):
    df = get_pandan()
    tidy = df.T.reset_index()
    tidy.columns.name = None
    tidy["Replicate"] = tidy["Replicate"].astype(int)
    meta_cols = ["Sample", "Replicate"]
    # Strip trailing or leading whitespace from column names
    tidy.columns = tidy.columns.str.strip()
    odour_cols = tidy.columns.difference(meta_cols)
    if do_zscore:
       tidy[odour_cols] = zscore(tidy[odour_cols].values, axis=1, nan_policy='omit')
    return tidy, odour_cols
