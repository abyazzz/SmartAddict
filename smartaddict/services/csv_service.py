import pandas as pd


GENDER_ALIASES = {
    "0": 0.0,
    "0.0": 0.0,
    "perempuan": 0.0,
    "female": 0.0,
    "wanita": 0.0,
    "p": 0.0,
    "1": 1.0,
    "1.0": 1.0,
    "laki-laki": 1.0,
    "laki laki": 1.0,
    "lakilaki": 1.0,
    "male": 1.0,
    "pria": 1.0,
    "l": 1.0,
}


def _normalize_text(value):
    return str(value).strip().lower().replace("_", " ")


def _parse_gender(value, row_index):
    key = _normalize_text(value)
    if key in GENDER_ALIASES:
        return GENDER_ALIASES[key]

    try:
        numeric_value = float(value)
    except (ValueError, TypeError):
        raise ValueError(
            f"Baris {row_index}, kolom gender harus berisi Perempuan/Laki-laki "
            "atau angka 0/1."
        )

    if numeric_value in (0, 1):
        return float(int(numeric_value))
    raise ValueError(f"Baris {row_index}, kolom gender hanya boleh Perempuan/Laki-laki.")


def _parse_feature_value(value, row_index, col_index):
    if pd.isna(value) or str(value).strip() == "":
        raise ValueError(f"Baris {row_index}, kolom {col_index} tidak boleh kosong.")
    if col_index == 2:
        return _parse_gender(value, row_index)
    try:
        return float(value)
    except (ValueError, TypeError):
        raise ValueError(f"Baris {row_index}, kolom {col_index} harus berupa angka.")


def _row_can_be_data(row):
    values = list(row.values)
    if len(values) not in (10, 11):
        return False
    try:
        for col_index, value in enumerate(values[:10], start=1):
            _parse_feature_value(value, 1, col_index)
        return True
    except ValueError:
        return False


def parse_csv_rows(file_obj):
    df_raw = pd.read_csv(file_obj, header=None, skipinitialspace=True)
    if len(df_raw) == 0:
        return [], False

    first_row = df_raw.iloc[0]
    has_header = not _row_can_be_data(first_row)

    df = df_raw.iloc[1:].reset_index(drop=True) if has_header else df_raw
    num_cols = len(df.columns)
    if num_cols == 11:
        df = df.iloc[:, :10]
    elif num_cols != 10:
        raise ValueError(f"CSV harus memiliki 10 kolom fitur (ditemukan {num_cols} kolom).")

    if len(df) == 0:
        return [], has_header
    if len(df) > 20:
        raise ValueError("CSV maksimal berisi 20 baris data.")

    rows = []
    for row_index, (_, row) in enumerate(df.iterrows(), start=1):
        values = []
        for col_index, value in enumerate(row.values, start=1):
            values.append(_parse_feature_value(value, row_index, col_index))
        rows.append(values)

    return rows, has_header


def average_rows(rows):
    if not rows:
        return []
    column_count = len(rows[0])
    return [sum(row[index] for row in rows) / len(rows) for index in range(column_count)]
