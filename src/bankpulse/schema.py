from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    age: float = Field(ge=18, le=100)
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    campaign: float = Field(ge=0)
    pdays: float = Field(ge=0)
    previous: float = Field(ge=0)
    poutcome: str
    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float


class PredictionResponse(BaseModel):
    probability: float = Field(ge=0, le=1)
    prediction: int = Field(ge=0, le=1)
    model_version: str
    latency_ms: float = Field(ge=0)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
