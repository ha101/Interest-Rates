Upload guide:

1) In your GitHub repo, create the folder path `.github/workflows` if it does not exist.
2) Upload the two YAML files from this `workflows/` folder into `.github/workflows/`:
   - prefetch_pages_data.yml
   - deploy_pages.yml
3) In GitHub settings:
   - Pages -> Build and deployment -> Source: GitHub Actions
   - Actions -> General -> Workflow permissions: Read and write permissions
4) Run the workflow `Prefetch rate data for Pages` once manually.
5) Confirm `Deploy GitHub Pages` runs successfully.
