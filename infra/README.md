# AWS setup files

These describe the AWS resources the pipeline runs on.

| File | What it is |
|---|---|
| `trust-ecs-tasks.json` | Lets AWS container tasks use the pipeline's two IAM roles |
| `s3-bucket-access.json` | The job role's permissions: read and write files in one bucket, nothing else |
| `compute-env.template.json` | The AWS Batch compute environment (Fargate Spot, up to 16 vCPUs) |
| `job-def.template.json` | The Batch job definition: which image to run, CPU and memory, and roles |
| `submit-train.template.json` | An example array job: 32 parallel chunks of 34 stars |

The `.template.json` files use placeholders such as `<ACCOUNT_ID>`, `<REGION>`, `<BUCKET>` and `<SUBNET_ID_1>`. Fill these in with your own values to recreate the setup. The filled-in copies are kept out of the repository by `.gitignore`, because they contain account-specific details.

These files will be replaced by Terraform in a later stage of the project.
