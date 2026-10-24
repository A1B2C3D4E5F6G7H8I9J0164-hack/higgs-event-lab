# Deployment Runbook

**Owner:** Krishna (Team Lead). **Scope:** Environment configuration, containerized deployment, smoke testing, and rollback procedure (`END-06`).

## 1. Prerequisites and Safety Rules

- Never commit raw datasets (`HIGGS.csv.gz`), model binaries (`*.pt`), or secrets to the repository.
- The dashboard container must start without triggering background model training.
- All visualizations load pre-generated experiment artifacts from `outputs/`.
- If no artifacts exist, the dashboard displays a clear empty state with reproduction instructions.

## 2. Local Containerized Deployment

### Build the Image
```bash
docker build -t higgs-event-lab:latest -f deployment/Dockerfile .
```

### Run the Container
```bash
docker run -d --name higgs-app -p 8501:8501 higgs-event-lab:latest
```

### Verify Container Health
```bash
curl -f http://localhost:8501/_stcore/health
```

Access the UI at `http://localhost:8501`.

### Stop and Remove
```bash
docker stop higgs-app && docker rm higgs-app
```

## 3. Cloud Deployment (Streamlit Community Cloud)

1. Connect the GitHub repository `CosmicMagnetar/higgs-event-lab`.
2. Set the main file path to `app/main.py`.
3. Set the Python version to 3.11+.
4. Ensure `PYTHONPATH=src` is defined if required by custom paths.
5. Deploy and verify the live public URL.

## 4. Smoke Test Checklist

- [ ] Container or web service starts up in less than 30 seconds.
- [ ] Health check endpoint (`/_stcore/health`) returns status 200.
- [ ] Title, scientific scope, and limitation notices render on the main page.
- [ ] Sidebar allows selecting model families and feature groups.
- [ ] Plots (ROC curves, PR curves, calibration) render interactively.
- [ ] Empty state renders gracefully when no artifacts are found.

## 5. Rollback Procedure

If a deployed version experiences unexpected regressions or runtime errors:

1. Identify the last verified stable release commit on `main`:
   ```bash
   git log --oneline -n 5
   ```
2. Revert to the stable commit or previous release tag:
   ```bash
   git checkout <stable-commit-hash>
   ```
3. Re-build and re-deploy the container:
   ```bash
   docker build -t higgs-event-lab:rollback -f deployment/Dockerfile .
   docker stop higgs-app && docker rm higgs-app
   docker run -d --name higgs-app -p 8501:8501 higgs-event-lab:rollback
   ```
4. Record the incident, root cause, and remediation steps in `team/BUG_TRACKER.md`.
