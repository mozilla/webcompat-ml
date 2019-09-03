# Machine learning pipeline

## Components

* Github repository `webcompat/web-bugs`
* Webhooks
* AWS Lambda (webhook handler)
* AWS API Gateway (HTTP endpoint)
* AWS Batch (workers)
* AWS S3 (storage)
* Docker (packaging)
* Web extension

## How it works

The source of truth for webcompat issues is the github repository `webcompat/web-bugs`. Every time we have activity in the repository issues, a webhook that starts the pipeline gets triggered.

The webhook is handled using an AWS Lambda function exposed using AWS API Gateway. There we verify the payload integrity to ensure that the request is valid. For every valid we request we submit a classification job in the task queue.

Our compute environment in AWS Batch picks up the tasks and runs them in servers with predefined quotas (min/max vCPU, min/max memory). The task definition is docker based. Each task is a command that is run using a predefined docker image. That means that we don't have any idling resources and we can scale up/down horizontally according to needs

Each task gets as input a Github issue ID and after fetching and classifying the issue, it stores the predictions as a JSON file in a public S3 bucket.

From there we can either use the web extension and get notifications while browsing Github or access the predictions directly using HTTP requests (eg `GET <bucket>.s3.<aws_region>.com/<issue_id>.json`).