from pydantic import BaseModel, Field
from typing import Any, Literal
import os

def load_config(filename) -> 'Config':
    with open(filename) as f:
        cfg = Config.model_validate_json(f.read())

    os.environ["BASE_PERIOD"] = str(cfg.LEVELS.BASE_PERIOD)
    os.environ["JAX_PLATFORMS"] = 'cpu'

    return cfg

class DatasetConfig(BaseModel):
    NAME: Literal['web-traffic', 'M5', 'M4'] = Field(alias='name')
    START: int = Field(alias='start', default=0)
    END: int = Field(alias='end', default=-1)

class AggregationsConfig(BaseModel):
    BASE_PERIOD: int = Field(alias='base_period')
    SUBSET: int = Field(alias='subset', default=0)
    DIFF: int = Field(alias='diff')
    MAX_H: int = Field(alias='max_height', default=0)
    SEQUENCES: int = Field(alias='sequences', default=10)

class SingleStepConfig(BaseModel):
    BASE_PERIOD: int = Field(alias='base_period')
    REPEATS: int = Field(alias='single_step_repeats')
    SUBSET: int = 0

    def model_post_init(self, context: Any) -> None:
        self.SUBSET = self.REPEATS

class Config(BaseModel):
    SEED: int = Field(alias='seed', default=42)
    ERROR_METRIC: str = Field(alias='error_metric')
    STATISTIC: str = Field(alias='statistic')
    VERBOSE: bool = Field(alias='verbose')
    N_JOBS: int = Field(alias='n_jobs')
    HORIZON_RATIO: int = Field(alias='horizon_ratio')
    MODEL: Literal['AutoETS', 'AutoARIMA', 'AutoTheta', 'Prophet'] = Field(alias='model')
    DATASET: DatasetConfig = Field(alias='dataset')
    LEVELS: AggregationsConfig | SingleStepConfig = Field(alias='levels')
    TEST: Literal['page l', 'slope t', 'wilcoxon', 'sign'] = Field(alias='test')
    HORIZON: int = Field(default=0)
    MASTER_FOLDER: str = Field(default='')
    FOLDER: str = Field(default='')

    def model_post_init(self, context: Any) -> None:
        self.HORIZON = self.LEVELS.BASE_PERIOD * self.HORIZON_RATIO
        if self.is_single_step_hierarchies():
            self.MASTER_FOLDER = f'period={self.LEVELS.BASE_PERIOD}_horizon={self.HORIZON}_single_step_repeats={self.LEVELS.SUBSET}'
        else:
            self.MASTER_FOLDER = f'period={self.LEVELS.BASE_PERIOD}_horizon={self.HORIZON}_subset={self.LEVELS.SUBSET}_d={self.LEVELS.DIFF}_maxh={self.LEVELS.MAX_H}_sequences={self.LEVELS.SEQUENCES}'
        self.FOLDER = f'{self.MASTER_FOLDER}/{self.DATASET.NAME}/{self.MODEL}'

    def is_single_step_hierarchies(self) -> bool:
        return isinstance(self.LEVELS, SingleStepConfig)