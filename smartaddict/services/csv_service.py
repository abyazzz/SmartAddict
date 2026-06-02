import pandas as pd


def parse_csv_rows(file_obj):
    df_raw = pd.read_csv(file_obj, header=None)
    if len(df_raw) == 0:
        return [], False

    first_row = df_raw.iloc[0]
    has_header = False
    try:
        [float(value) for value in first_row]
    except (ValueError, TypeError):
        has_header = True

    df = df_raw.iloc[1:].reset_index(drop=True) if has_header else df_raw
    num_cols = len(df.columns)
    if num_cols == 11:
        df = df.iloc[:, :10]
    elif num_cols != 10:
        raise ValueError(f"CSV harus memiliki 10 kolom fitur (ditemukan {num_cols} kolom).")

    if len(df) == 0:
        return [], False
    if len(df) > 20:
        raise ValueError("CSV maksimal berisi 20 baris data.")

    rows = []
    for row_index, (_, row) in enumerate(df.iterrows(), start=1):
        values = []
        for col_index, value in enumerate(row.values, start=1):
            if pd.isna(value):
                raise ValueError(f"Baris {row_index}, kolom {col_index} tidak boleh kosong.")
            values.append(float(value))
        rows.append(values)

    return rows, has_header


def average_rows(rows):
    if not rows:
        return []
    column_count = len(rows[0])
    return [sum(row[index] for row in rows) / len(rows) for index in range(column_count)]