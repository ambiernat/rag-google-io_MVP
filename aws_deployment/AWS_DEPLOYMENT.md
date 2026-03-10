# Deploying a RAG API to AWS with ECS Fargate

This document walks through the end-to-end process of containerising the RAG Google I/O API and deploying it to AWS using ECR, ECS, and Fargate — with no servers to manage.

---

## Architecture Overview

```
Local Code (FastAPI + Qdrant)
        ↓
   Docker Image
        ↓
  EC2 (docker build)
        ↓
ECR (image registry)
        ↓
ECS Task Definition (blueprint)
        ↓
ECS Service → Fargate (serverless compute)
        ↓
  Public IP via VPC
        ↓
  CloudWatch Logs
```

The app runs as two containers in a single Fargate task: **FastAPI** (3 GB RAM) serving the retrieval API, and **Qdrant** (1 GB RAM) as the vector store sidecar.

---

## Step 1 — Containerise the App (Docker)

The `Dockerfile` bundles the Python environment, FastAPI app, and all dependencies into a single image. This ensures the app runs identically in any environment.

CloudShell was the first attempt at building the image, but ran out of disk space on larger builds — which led to spinning up an EC2 instance instead (see Step 3).

---

## Step 2 — Create an ECR Repository

ECR (Elastic Container Registry) is AWS's private Docker image registry. The repository `fastapi-rag` was created to store the image before ECS can pull it.

From the ECR console, the **View push commands** dialog provides the exact CLI commands needed to authenticate, tag, and push the image:

![ECR push commands](01_ecr_push_commands.png)

The repository URI is:
```
886166401772.dkr.ecr.us-east-1.amazonaws.com/fastapi-rag:latest
```

---

## Step 3 — Build the Docker Image on EC2

CloudShell has a disk limit that makes building large images impractical. A `t3.small` EC2 instance was launched temporarily just to build and push the image.

**Launch the EC2 instance:**

![EC2 launch success](screenshots/08_ec2_launch_success.png)

**Connect via EC2 Instance Connect and run the build:**

```bash
# Configure AWS CLI
aws configure
# Clone the repo
git clone https://github.com/ambiernat/rag-google-io_v2.git
cd rag-google-io_v2
# Build the Docker image
docker build -t fastapi-rag .
```

![EC2 SSH build](screenshots/09_ec2_connect_build.png)

The build process downloads and compiles all Python dependencies:

![Docker build layers](screenshots/02_docker_build_ec2.png)

Once built, authenticate to ECR and push:

```bash
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin \
    886166401772.dkr.ecr.us-east-1.amazonaws.com

docker tag fastapi-rag:latest \
  886166401772.dkr.ecr.us-east-1.amazonaws.com/fastapi-rag:latest

docker push \
  886166401772.dkr.ecr.us-east-1.amazonaws.com/fastapi-rag:latest
```

![ECR login succeeded + repo details](screenshots/03_ecr_repo_login_success.png)

The 550 MB image is now stored in ECR. The EC2 instance was terminated immediately after — no ongoing cost.

---

## Step 4 — Create an ECS Cluster

ECS (Elastic Container Service) is the container orchestrator. A **cluster** is a logical grouping for tasks and services.

Starting from an empty cluster list:

![Empty ECS clusters](screenshots/04_ecs_clusters_empty.png)

Create a new cluster named `rag-cluster`, selecting **Fargate only** as the compute type (serverless — no EC2 instances to manage):

![Create ECS cluster](screenshots/05_ecs_create_cluster.png)

---

## Step 5 — Create a Task Definition

A **task definition** is the blueprint that tells ECS what containers to run, how much memory and CPU to allocate, and which image to use.

Configuration for `rag-task`:
- **Launch type:** AWS Fargate
- **CPU:** 1 vCPU
- **Memory:** 4 GB
- **Containers:** 2 — `fastapi` (3 GB) and `qdrant` (1 GB)

![Create task definition](screenshots/06_ecs_task_definition.png)

The FastAPI container image points to the ECR URI pushed in Step 3. The Qdrant container uses the public `qdrant/qdrant:latest` image.

---

## Step 6 — Create a Service with Networking

An ECS **service** keeps a specified number of task instances running at all times. Creating the service also sets up the VPC networking and security group.

A new security group `rag-sg` was configured with an inbound rule allowing TCP traffic on port **8000** from anywhere (`0.0.0.0/0`), so the FastAPI docs and API are publicly accessible:

![Security group port 8000](screenshots/07_ecs_security_group_port8000.png)

The service `rag-task3` was created with **desired count = 1** (one running task).

---

## Step 7 — Task Running on Fargate

Once deployed, ECS shows the task status as **Running** under the `rag-task3` service. Both containers — `qdrant` and `fastapi` — are reported as healthy:

![ECS task running](10_ecs_task_running.png)

The task configuration panel shows the Fargate compute details and the **public IP** assigned by the VPC:

![Fargate public IP](11_fargate_public_ip.png)

The app is accessible at `http://52.90.31.249:8000`.

---

## Step 8 — Verify the Deployment

**Health check** — confirms the FastAPI app is running and responsive:

![Health endpoint returning ok](12_health_endpoint_ok.png)

**API docs** — the full Swagger UI confirms all three retrieval endpoints (dense, sparse, hybrid) are live:

![API docs live](13_api_docs_live.png)

---

## Step 9 — CloudWatch Logs

All container stdout is streamed to CloudWatch under the log group `/ecs/rag-task`. This was used during debugging to catch errors like missing BM25 model files, which were fixed before the final successful deployment.

Cost: ~$0.50/month for log storage.

---

## Step 10 — Scale to Zero (Cost Control)

With no load balancer or reserved capacity, the only ongoing costs when the app is running are Fargate compute (~$42/month for 1 vCPU + 4 GB). To pause the app without destroying any infrastructure, the service desired count was set to 0:

```bash
aws ecs update-service \
  --cluster rag-cluster \
  --service rag-task3 \
  --desired-count 0 \
  --region us-east-1
```

To restart:

```bash
aws ecs update-service \
  --cluster rag-cluster \
  --service rag-task3 \
  --desired-count 1 \
  --region us-east-1
```

Wait 2–3 minutes, then retrieve the new public IP from the ECS console.

---

## Cost Summary

| Service | Status | Monthly Cost |
|---|---|---|
| ECR | Image stored (550 MB) | ~$0.06 |
| ECS | Service exists, 0 tasks running | $0 |
| Fargate | Not running (desired count = 0) | $0 |
| CloudWatch | Logs stored | ~$0.50 |
| EC2 | Terminated | $0 |
| **Total** | **Paused** | **~$0.56** |

When running: **~$42/month**.

---

## IAM Setup

Two IAM principals were used:

- **`al-deploy`** — IAM user with ECR Full Access and ECS Full Access, used for CLI operations
- **`ecsTaskExecutionRole`** — IAM role attached to the task definition, allowing ECS to pull images from ECR and write logs to CloudWatch

---

## Key Learnings

- **CloudShell disk limits** make it unsuitable for building large Docker images. Use a temporary EC2 instance instead — it's cheap and terminates cleanly.
- **Two-container task definitions** work well for a FastAPI + vector DB pattern. Qdrant runs as a sidecar on `localhost` within the same task networking namespace.
- **Fargate public IPs are ephemeral** — each new task gets a different IP. For production, add an Application Load Balancer with a fixed DNS name.
- **Scale to 0** is the right default for development/portfolio deployments — all infrastructure stays configured, costs drop to nearly zero.
