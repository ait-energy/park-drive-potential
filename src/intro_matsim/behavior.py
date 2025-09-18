import math
from collections.abc import Collection
from collections import namedtuple

from matplotlib import pyplot as plt
import pandas as pd

MODEL_NONE = "NONE"
MODEL_DOMINO = "DOMINO"
MODEL_PNM = "PNM"

MODE_COMBINED = "combined"
MODE_DRIVER = "driver"
MODE_PASSENGER = "passenger"

ModelInfo = namedtuple("ModelInfo", ["key", "name", "color_dark", "color_light"])
cmap = plt.get_cmap("Paired")
MODELS = {
    MODEL_NONE: ModelInfo(MODEL_NONE, "ohne", cmap(1), cmap(0)),
    MODEL_DOMINO: ModelInfo(MODEL_DOMINO, "DOMINO", cmap(9), cmap(8)),
    MODEL_PNM: ModelInfo(MODEL_PNM, "pro:NEWmotion", cmap(5), cmap(4)),
}
"""dict of model ids and infos about them. keys sorted by model specificity"""


_MAX_AGE = 99


def model_id(model: str, mode: str) -> str:
    return f"bm.{model}.{mode}"


def parse_model_id(model_id: str) -> tuple[str, str]:
    parts = model_id.split(".")
    if len(parts) != 3 or not parts[0] == "bm":
        raise ValueError(f"Invalid model ID: {model_id}")
    return parts[1], parts[2]


class BehaviorModel:
    """
    Park and Drive behavior model based on pro:NEWmotion and DOMINO survey data.
    The model is used to determine the probability of a person using park and drive.
    """

    def __init__(self):
        pnm = BehaviorModel._read_csv("PG_agg_MATSim/pnm_Survey/agg_pg_final_perc.csv")
        self.models = {
            MODEL_PNM: {
                MODE_COMBINED: pnm,
                MODE_DRIVER: pnm,
                MODE_PASSENGER: pnm,
            },
            MODEL_DOMINO: {
                MODE_COMBINED: BehaviorModel._read_csv(
                    "PG_agg_MATSim/DOMINO_Survey/agg_pg_final_perc_all.csv"
                ),
                MODE_DRIVER: self._read_csv(
                    "PG_agg_MATSim/DOMINO_Survey/agg_pg_final_perc_driver.csv"
                ),
                MODE_PASSENGER: BehaviorModel._read_csv(
                    "PG_agg_MATSim/DOMINO_Survey/agg_pg_final_perc_passenger.csv"
                ),
            },
        }
        self.age_map = {}
        for model, mode in self.models.items():
            for mode, df in mode.items():
                for job in df.index.get_level_values("job"):
                    age_categories = df.loc[job].index.get_level_values("age")
                    self.age_map.setdefault(model, {}).setdefault(mode, {})[job] = (
                        BehaviorModel._int_to_category_map(age_categories)
                    )

    @staticmethod
    def _read_csv(file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        df = df.set_index(["job", "age"])
        return df

    @staticmethod
    def _int_to_category_map(categories: Collection[str]) -> dict[int, str]:
        int_to_cat = {}
        for category in categories:
            cat = category.replace("Jahre", "").strip()
            if "-" in cat:
                start, end = map(int, cat.split("-"))
                for i in range(start, end + 1):
                    int_to_cat[i] = category
            elif "+" in cat:
                start = int(cat.replace("+", ""))
                for i in range(start, _MAX_AGE + 1):
                    int_to_cat[i] = category
            else:
                raise ValueError(f"Unknown age category: {category}")
        return int_to_cat

    def get_choice_probability(
        self,
        model_name: str,
        mode: str,
        employed: bool,
        full_time: bool,
        in_education: bool,
        age: int,
    ) -> float:
        """
        Get the probability (between 0 and 1) that a person will use park and drive
        """
        try:
            job_cat = BehaviorModel._job_category(employed, full_time, in_education)
            age_cat = self.age_map[model_name][mode][job_cat][
                age if age < _MAX_AGE else _MAX_AGE
            ]
            res = self.models[model_name][mode].loc[(job_cat, age_cat), "Perc_Yes"]
            return float(res)  # type: ignore
        except KeyError:
            return math.nan

    @staticmethod
    def _job_category(employed: bool, full_time: bool, in_education: bool) -> str:
        if employed:
            return "Berufstätig, Vollzeit" if full_time else "Berufstätig, Teilzeit"
        elif in_education:
            return "in Ausbildung"
        return "Nicht berufstätig"
