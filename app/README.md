# Dashboard

Ronit owns the Streamlit UI. It should read saved artifacts created by experiment runs, display data provenance and model metadata, compare models/features, show ROC/PR/calibration and the validation-selected operating point, and export a result summary.

The app must have an honest empty state when outputs are missing. It must not trigger expensive training on startup or display hard-coded benchmark results. Krishna owns deployment and production configuration.