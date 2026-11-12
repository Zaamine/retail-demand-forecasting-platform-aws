"""Render AWS operations IaC. No AWS resources are modified."""

import json


def state_machine_arn(name):
    return (
        "arn:${AWS::Partition}:states:${AWS::Region}:"
        "${AWS::AccountId}:stateMachine:" + name
    )


def build_template():
    workflows = {
        "v1": "retail-v1-forecast-workflow",
        "v2": "retail-v2-test-forecast-workflow",
    }

    topic = {"Ref": "AlertsTopic"}

    resources = {
        "AlertsTopic": {
            "Type": "AWS::SNS::Topic",
            "Properties": {
                "TopicName": "retail-forecast-operations-alerts"
            },
        },
        "TopicPolicy": {
            "Type": "AWS::SNS::TopicPolicy",
            "Properties": {
                "Topics": [topic],
                "PolicyDocument": {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Sid": "AccountAdministration",
                            "Effect": "Allow",
                            "Principal": {
                                "AWS": {
                                    "Fn::Sub": (
                                        "arn:${AWS::Partition}:iam::"
                                        "${AWS::AccountId}:root"
                                    )
                                }
                            },
                            "Action": [
                                "sns:GetTopicAttributes",
                                "sns:SetTopicAttributes",
                                "sns:AddPermission",
                                "sns:RemovePermission",
                                "sns:DeleteTopic",
                                "sns:Subscribe",
                                "sns:ListSubscriptionsByTopic",
                                "sns:Publish",
                            ],
                            "Resource": topic,
                        },
                        {
                            "Sid": "CloudWatchAlarmNotifications",
                            "Effect": "Allow",
                            "Principal": {
                                "Service": "cloudwatch.amazonaws.com"
                            },
                            "Action": "sns:Publish",
                            "Resource": topic,
                            "Condition": {
                                "ArnLike": {
                                    "aws:SourceArn": {
                                        "Fn::Sub": (
                                            "arn:${AWS::Partition}:cloudwatch:"
                                            "${AWS::Region}:${AWS::AccountId}:"
                                            "alarm:retail-forecast-*"
                                        )
                                    }
                                }
                            },
                        },
                    ],
                },
            },
        },
    }

    # Alert on failures, timeouts and aborted executions.
    metrics = (
        "ExecutionsFailed",
        "ExecutionsTimedOut",
        "ExecutionsAborted",
    )

    for version, name in workflows.items():
        arn = state_machine_arn(name)

        for metric in metrics:
            logical_id = version.upper() + metric

            resources[logical_id] = {
                "Type": "AWS::CloudWatch::Alarm",
                "Properties": {
                    "AlarmName": (
                        f"retail-forecast-{version}-{metric.lower()}"
                    ),
                    "AlarmDescription": (
                        f"{name}: {metric} detected"
                    ),
                    "Namespace": "AWS/States",
                    "MetricName": metric,
                    "Dimensions": [
                        {
                            "Name": "StateMachineArn",
                            "Value": {"Fn::Sub": arn},
                        }
                    ],
                    "Statistic": "Sum",
                    "Period": 300,
                    "EvaluationPeriods": 1,
                    "Threshold": 1,
                    "ComparisonOperator": (
                        "GreaterThanOrEqualToThreshold"
                    ),
                    "TreatMissingData": "notBreaching",
                    "AlarmActions": [topic],
                },
            }

    widgets = []

    for index, (version, name) in enumerate(workflows.items()):
        arn = state_machine_arn(name)

        chart_metrics = [
            [
                "AWS/States",
                metric,
                "StateMachineArn",
                arn,
                {"label": metric},
            ]
            for metric in (
                "ExecutionsStarted",
                "ExecutionsSucceeded",
                *metrics,
            )
        ]

        widgets.append({
            "type": "metric",
            "x": index * 12,
            "y": 0,
            "width": 12,
            "height": 7,
            "properties": {
                "title": f"{version.upper()} — {name}",
                "view": "timeSeries",
                "region": "${AWS::Region}",
                "stat": "Sum",
                "period": 300,
                "metrics": chart_metrics,
            },
        })

    resources["MonitoringDashboard"] = {
        "Type": "AWS::CloudWatch::Dashboard",
        "Properties": {
            "DashboardName": "RetailForecast-Week4B-Operations",
            "DashboardBody": {
                "Fn::Sub": json.dumps({
                    "widgets": [
                        *widgets,
                        {
                            "type": "text",
                            "x": 0,
                            "y": 7,
                            "width": 24,
                            "height": 2,
                            "properties": {
                                "markdown": (
                                    "**Deployed Git revision:** "
                                    "`${DeploymentRevision}`"
                                )
                            }
                        }
                    ]
                })
            },
        },
    }

    # The scheduler is deliberately disabled.
    resources["SchedulerRole"] = {
        "Type": "AWS::IAM::Role",
        "Properties": {
            "RoleName": "RetailForecastWeek4BSchedulerRole",
            "AssumeRolePolicyDocument": {
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "scheduler.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole",
                }],
            },
            "Policies": [{
                "PolicyName": "StartV2Only",
                "PolicyDocument": {
                    "Version": "2012-10-17",
                    "Statement": [{
                        "Effect": "Allow",
                        "Action": "states:StartExecution",
                        "Resource": {
                            "Fn::Sub": state_machine_arn(
                                workflows["v2"]
                            )
                        },
                    }],
                },
            }],
        },
    }

    resources["DisabledSchedule"] = {
        "Type": "AWS::Scheduler::Schedule",
        "Properties": {
            "Name": "retail-v2-demo-daily-disabled",
            "Description": (
                "Historical 2017 data: do not enable "
                "without a fresh data source"
            ),
            "State": "DISABLED",
            "ScheduleExpression": "cron(0 5 * * ? *)",
            "ScheduleExpressionTimezone": "Etc/UTC",
            "FlexibleTimeWindow": {"Mode": "OFF"},
            "Target": {
                "Arn": {
                    "Fn::Sub": state_machine_arn(workflows["v2"])
                },
                "RoleArn": {
                    "Fn::GetAtt": ["SchedulerRole", "Arn"]
                },
                "Input": json.dumps({
                    "run_id": "<aws.scheduler.execution-id>"
                }),
            },
        },
    }

    return {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Description": (
            "Retail forecasting monitoring, SNS alerting, "
            "and disabled EventBridge demo schedule."
        ),
        "Parameters": {
            "DeploymentRevision": {
                "Type": "String",
                "Default": "local",
                "AllowedPattern": "[A-Za-z0-9._-]+"
            }
        },
        "Resources": resources,
        "Outputs": {
            "DeploymentRevision": {
                "Value": {"Ref": "DeploymentRevision"}
            },
            "AlertsTopicArn": {"Value": topic},
            "DashboardName": {
                "Value": "RetailForecast-Week4B-Operations"
            },
            "SchedulingStatus": {"Value": "DISABLED"},
        },
    }


if __name__ == "__main__":
    print(json.dumps(build_template(), indent=2))
