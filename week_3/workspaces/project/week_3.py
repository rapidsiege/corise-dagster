from typing import List

from dagster import (
    In,
    Out,
    ResourceDefinition,
    String,
    RetryPolicy,
    RunRequest,
    ScheduleDefinition,
    SkipReason,
    graph,
    op,
    schedule,
    sensor,
    static_partitioned_config,
)
from workspaces.config import REDIS, S3
from workspaces.project.sensors import get_s3_keys
from workspaces.resources import mock_s3_resource, redis_resource, s3_resource
from workspaces.types import Aggregation, Stock


@op(
    description="Returns list of stock data from s3_key",
    config_schema={"s3_key": str},
    required_resource_keys={"s3"},
    out={"stocks": Out(dagster_type=List[Stock], description="List of stocks")},
    tags={"kind": "s3"},
)
def get_s3_data(context) -> List[Stock]:
    file_name = context.op_config["s3_key"]
    stocks = context.resources.s3.get_data(file_name)
    return list(map(Stock.from_list, stocks))


@op(
    description="Returns the Aggregation from the stock value with the greatest high value",
    ins={"stocks": In(dagster_type=List[Stock], description="List of stocks")},
    out={
        "stock_greatest_high": Out(
            dagster_type=Aggregation,
            description="Date and high values from the stock value with the greatest high value",
        )
    },
)
def process_data(context, stocks) -> Aggregation:
    max_high_value = max(stocks, key=lambda x: x.high)
    return Aggregation(date=max_high_value.date, high=max_high_value.high)


@op(
    required_resource_keys={"redis"},
    description="Uploads stock_greatest_high (Aggregation class type) to Redis",
    ins={
        "stock_greatest_high": In(
            dagster_type=Aggregation,
            description="Date and high values from the stock value with the greatest high value",
        )
    },
    tags={"kind": "redis"},
)
def put_redis_data(context, stock_greatest_high):
    context.resources.redis.put_data(
        name=String(stock_greatest_high.date), value=String(stock_greatest_high.high)
    )


@op(
    required_resource_keys={"s3"},
    description="Upload stock_greatest_high (Aggregation class type) to S3 data lake",
    ins={
        "stock_greatest_high": In(
            dagster_type=Aggregation,
            description="Date and high values from the stock value with the greatest high value",
        )
    },
    tags={"kind": "s3"},
)
def put_s3_data(context, stock_greatest_high):
    context.resources.s3.put_data(
        key_name=String(stock_greatest_high.date), data=stock_greatest_high
    )


@graph
def machine_learning_graph():
    data = get_s3_data()
    highest_data = process_data(data)
    put_redis_data(highest_data)
    put_s3_data(highest_data)


local = {
    "ops": {"get_s3_data": {"config": {"s3_key": "prefix/stock_1.csv"}}},
}


@static_partitioned_config(partition_keys=[str(n) for n in range(1, 11)])
def docker_config(partition_key: String):
    return {
        "resources": {
            "s3": {"config": S3},
            "redis": {"config": REDIS},
        },
        "ops": {
            "get_s3_data": {"config": {"s3_key": f"prefix/stock_{partition_key}.csv"}}
        },
    }


machine_learning_job_local = machine_learning_graph.to_job(
    name="machine_learning_job_local",
    config=local,
    resource_defs={"s3": mock_s3_resource, "redis": ResourceDefinition.mock_resource()},
)

machine_learning_job_docker = machine_learning_graph.to_job(
    name="machine_learning_job_docker",
    config=docker_config,
    resource_defs={"s3": s3_resource, "redis": redis_resource},
    op_retry_policy=RetryPolicy(max_retries=10, delay=1),
)


machine_learning_schedule_local = ScheduleDefinition(
    job=machine_learning_job_local,
    cron_schedule="*/15 * * * *",
)


@schedule(job=machine_learning_job_docker, cron_schedule="0 * * * *")
def machine_learning_schedule_docker():
    for partition_key in docker_config.get_partition_keys():
        yield RunRequest(
            run_key=partition_key,
            run_config=docker_config.get_run_config_for_partition_key(partition_key),
        )


@sensor(job=machine_learning_job_docker)
def machine_learning_sensor_docker(context):
    keys_found = get_s3_keys(
        bucket="dagster",
        prefix="prefix",
        endpoint_url="http://localstack:4566",
    )
    if not keys_found:
        yield SkipReason("No new s3 files found in bucket.")
        return
    for key_found in keys_found:
        yield RunRequest(
            run_key=key_found,
            run_config={
                "resources": {
                    "s3": {"config": S3},
                    "redis": {"config": REDIS},
                },
                "ops": {
                    "get_s3_data": {
                        "config": {"s3_key": key_found},
                    },
                },
            },
        )
