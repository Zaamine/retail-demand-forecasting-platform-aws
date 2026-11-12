"""Pure pandas validation of frozen V1 prediction business keys."""
import pandas as pd

KEYS = ["store_nbr", "family", "forecast_origin", "horizon", "target_date"]

def align_forecasts(produced, reference):
    """Enforce complete one-to-one matching of the five business keys."""
    left, right = produced.copy(), reference.copy()
    for frame in (left, right):
        frame["family"] = frame["family"].astype("string")
        for column in ("forecast_origin", "target_date"):
            frame[column] = pd.to_datetime(frame[column])
        if frame.duplicated(KEYS).any():
            raise ValueError("Duplicate forecast business keys")
    compared = left.merge(
        right[KEYS + ["prediction_raw", "prediction"]],
        on=KEYS, how="outer", suffixes=("_aws", "_notebook"),
        validate="one_to_one", indicator=True,
    )
    if not compared["_merge"].eq("both").all():
        raise ValueError("Batch predictions do not cover exactly the frozen notebook keys")
    return compared.drop(columns=["_merge"])
