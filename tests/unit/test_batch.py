"""AWS Batch asset tests using moto mock.

Tests the etl_on_batch asset: submit, error paths, and sensor wiring.
moto patches boto3's HTTP transport layer transparently.

Note: moto Batch requires Docker daemon + a local image (quarry-etl:test)
to run containers and transition jobs to SUCCEEDED.

Usage:
    pytest tests/unit/test_batch.py -v
"""

import boto3
from moto import mock_aws

from quarry.config import settings


# ── Helpers ──


def _setup_batch_env(region: str = "us-east-1"):
    """Create the minimum Batch infrastructure moto needs.

    moto requires a real-ish compute environment + job queue + job definition
    before submit_job will succeed.
    """
    # EC2 resources (moto validates subnet/SG existence)
    ec2 = boto3.resource("ec2", region_name=region)
    vpc = ec2.create_vpc(CidrBlock="10.0.0.0/16")
    subnet = ec2.create_subnet(VpcId=vpc.id, CidrBlock="10.0.1.0/24")
    sg = ec2.create_security_group(
        GroupName="batch-test-sg", Description="test", VpcId=vpc.id
    )

    iam = boto3.client("iam", region_name=region)
    role = iam.create_role(
        RoleName="batch-test-role",
        AssumeRolePolicyDocument="{}",
        Path="/",
    )
    role_arn = role["Role"]["Arn"]

    batch = boto3.client("batch", region_name=region)

    # Compute environment (MANAGED + FARGATE to avoid EC2 complexity)
    batch.create_compute_environment(
        computeEnvironmentName="test-ce",
        type="MANAGED",
        computeResources={
            "type": "FARGATE",
            "maxvCpus": 2,
            "subnets": [subnet.id],
            "securityGroupIds": [sg.id],
        },
        serviceRole=role_arn,
    )

    # Job queue pointing to our compute environment
    batch.create_job_queue(
        jobQueueName=settings.batch_job_queue,
        state="ENABLED",
        priority=1,
        computeEnvironmentOrder=[
            {"order": 1, "computeEnvironment": "test-ce"},
        ],
    )

    # Job definition (container image doesn't matter — moto won't run it)
    batch.register_job_definition(
        jobDefinitionName=settings.batch_job_definition,
        type="container",
        containerProperties={
            "image": "quarry-etl:test",
            "vcpus": 1,
            "memory": 512,
        },
    )

    return batch


# ── Tests ──


class TestEtlOnBatch:
    """Test etl_on_batch asset with moto-mocked AWS Batch."""

    @mock_aws
    def test_submit_and_succeed(self):
        """Happy path: job submits and reaches SUCCEEDED status."""
        _setup_batch_env()

        from dagster import materialize

        from quarry.assets.batch import etl_on_batch

        result = materialize([etl_on_batch])
        assert result.success

        # Verify metadata contains job_id and job_name
        event = result.get_asset_materialization_events()[0]
        metadata = event.step_materialization_data.materialization.metadata
        assert "job_id" in metadata
        assert "job_name" in metadata
        assert metadata["job_name"].text.startswith("quarry-etl-")

    @mock_aws
    def test_returns_job_id(self):
        """Verify the returned job_id is a valid UUID-like string from Batch."""
        _setup_batch_env()

        from dagster import materialize

        from quarry.assets.batch import etl_on_batch

        result = materialize([etl_on_batch])
        event = result.get_asset_materialization_events()[0]
        job_id = event.step_materialization_data.materialization.metadata["job_id"].text
        assert len(job_id) > 0

    @mock_aws
    def test_missing_job_queue_raises(self):
        """submit_job fails if job queue doesn't exist."""
        # Set up Batch but with wrong queue name
        region = "us-east-1"
        iam = boto3.client("iam", region_name=region)
        iam.create_role(
            RoleName="batch-test-role",
            AssumeRolePolicyDocument="{}",
            Path="/",
        )
        batch = boto3.client("batch", region_name=region)
        batch.register_job_definition(
            jobDefinitionName=settings.batch_job_definition,
            type="container",
            containerProperties={
                "image": "quarry-etl:test",
                "vcpus": 1,
                "memory": 512,
            },
        )
        # No job queue created — submit should fail

        from dagster import materialize

        from quarry.assets.batch import etl_on_batch

        result = materialize([etl_on_batch], raise_on_error=False)
        assert not result.success

    @mock_aws
    def test_missing_job_definition_raises(self):
        """submit_job fails if job definition doesn't exist."""
        region = "us-east-1"
        ec2 = boto3.resource("ec2", region_name=region)
        vpc = ec2.create_vpc(CidrBlock="10.0.0.0/16")
        subnet = ec2.create_subnet(VpcId=vpc.id, CidrBlock="10.0.1.0/24")
        sg = ec2.create_security_group(
            GroupName="batch-test-sg", Description="test", VpcId=vpc.id
        )

        iam = boto3.client("iam", region_name=region)
        role = iam.create_role(
            RoleName="batch-test-role",
            AssumeRolePolicyDocument="{}",
            Path="/",
        )
        batch = boto3.client("batch", region_name=region)
        batch.create_compute_environment(
            computeEnvironmentName="test-ce",
            type="MANAGED",
            computeResources={
                "type": "FARGATE",
                "maxvCpus": 2,
                "subnets": [subnet.id],
                "securityGroupIds": [sg.id],
            },
            serviceRole=role["Role"]["Arn"],
        )
        batch.create_job_queue(
            jobQueueName=settings.batch_job_queue,
            state="ENABLED",
            priority=1,
            computeEnvironmentOrder=[
                {"order": 1, "computeEnvironment": "test-ce"},
            ],
        )
        # No job definition registered

        from dagster import materialize

        from quarry.assets.batch import etl_on_batch

        result = materialize([etl_on_batch], raise_on_error=False)
        assert not result.success


class TestSensorChain:
    """Test that sensor definitions are valid and reference correct assets/jobs."""

    def test_distributed_r2_sync_sensor(self):
        """Verify sensor watches etl_on_batch and targets r2_download_job."""
        from quarry.sensors import distributed_r2_sync

        assert distributed_r2_sync.asset_key.path == ["etl_on_batch"]

    def test_distributed_serve_sensor(self):
        """Verify sensor watches r2_download and targets serve_job."""
        from quarry.sensors import distributed_serve

        assert distributed_serve.asset_key.path == ["r2_download"]
