# Deployment

Krishna owns deployment, environment configuration, release and rollback. Deployment starts only after the real artifact schema is stable, the app loads artifacts without training on startup, smoke tests pass, no secrets are tracked, and the limitation statement is visible.

The provider/platform will be selected later based on the team's course requirements, account limitations, resource needs and availability. Do not put a real secret in this file; use platform environment settings and local `.env` files that are excluded by Git.