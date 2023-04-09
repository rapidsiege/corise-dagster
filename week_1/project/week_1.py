import csv
from datetime import datetime
from typing import Iterator, List

from dagster import (
    In,
    Nothing,
    OpExecutionContext,
    Out,
    String,
    job,
    op,
    usable_as_dagster_type,
)
from pydantic import BaseModel


@usable_as_dagster_type(description="Stock data")
class Stock(BaseModel):
    date: datetime
    close: float
    volume: int
    open: float
    high: float
    low: float

    @classmethod
    def from_list(cls, input_list: List[str]):
        """Do not worry about this class method for now"""
        return cls(
            date=datetime.strptime(input_list[0], "%Y/%m/%d"),
            close=float(input_list[1]),
            volume=int(float(input_list[2])),
            open=float(input_list[3]),
            high=float(input_list[4]),
            low=float(input_list[5]),
        )


@usable_as_dagster_type(description="Aggregation of stock data")
class Aggregation(BaseModel):
    date: datetime
    high: float


def csv_helper(file_name: str) -> Iterator[Stock]:
    with open(file_name) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            yield Stock.from_list(row)


@op(
    description = "Returns list of stock data from s3_key",
    config_schema = {"s3_key": str},
    out = {"stocks": Out(dagster_type = List[Stock],
        description = "List of stocks")}
)
def get_s3_data_op(context) -> List[Stock]:
    file_name = context.op_config["s3_key"]
    return list(csv_helper(file_name))


@op(
    description = "Returns the Aggregation from the stock value with the greatest high value",
    ins = {"stocks": In(dagster_type = List[Stock],
        description = "List of stocks")},
    out = {"stock_greatest_high": Out(dagster_type = Aggregation,
        description = "Date and high values from the stock value with the greatest high value")}
)
def process_data_op(context, stocks) -> Aggregation:
    max_high_value = max(stocks, key = lambda x: x.high)
    return Aggregation(date = max_high_value.date, high = max_high_value.high)


@op(
    description = "Currently unused. Will upload stock_greatest_high (Aggregation class type) to Redis",
    ins = {"stock_greatest_high": In(dagster_type = Aggregation,
        description = "Date and high values from the stock value with the greatest high value")}
)
def put_redis_data_op(context, stock_greatest_high) -> Nothing:
    pass


@op(
    description = "Currently unused. Will upload stock_greatest_high (Aggregation class type) to s3 data lake",
    ins = {"stock_greatest_high": In(dagster_type = Aggregation,
        description = "Date and high values from the stock value with the greatest high value")}
)
def put_s3_data_op(context, stock_greatest_high) -> Nothing:
    pass


@job(
    config = {"ops": {"get_s3_data_op": {"config": {"s3_key": "week_1/data/stock.csv"}}}},
    description = "Currently passing .csv filepath as s3_key so that the job works in dagit"
)
def machine_learning_job():
    put_redis_data_op(process_data_op(get_s3_data_op()))
    ##put_s3_data_op(process_data_op(get_s3_data_op()))
