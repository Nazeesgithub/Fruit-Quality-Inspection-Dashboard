# Deploy to Fly.io

## Prerequisites
1. Create free account at https://fly.io/
2. Install Fly CLI: https://fly.io/docs/getting-started/installing-flyctl/
3. Authenticate with Fly.io: `flyctl auth login`

## Step 1: Update App Name
Edit `fly.toml` and change `app = "fruit-quality-inspection"` to your desired app name (e.g., `fruit-classifier-demo`).

## Step 2: Deploy
Run from project root:
```bash
flyctl deploy
```

First deployment takes 5-10 minutes. Subsequent deploys are faster.

## Step 3: View Your App
```bash
flyctl open
```

Your app will be live at: `https://your-app-name.fly.dev`

## Step 4: View Logs
```bash
flyctl logs
```

## Troubleshooting

### Out of Memory
Edit `fly.toml` and increase memory:
```toml
[[vm]]
  memory_mb = 2048
```
Then redeploy: `flyctl deploy`

### Model Files Not Found
The model files must be in the `models/` directory before deployment. Ensure:
- `models/fruit_mobilenetv2.keras` exists
- `models/labels.json` exists

Or the app will show: "Model not found. Train first..."

### Scale Down (Save Credits)
```bash
flyctl scale vm 1
```

### Delete App
```bash
flyctl apps destroy your-app-name
```

## Notes
- Free tier: $5/month always free, additional usage billed separately
- You get 3 shared-cpu-1x 256MB VMs free every month
- Your app is in region `iad` (Virginia, USA)—change in `fly.toml` if needed

## Regions Available
- `iad` (Virginia, USA)
- `lhr` (London, UK)
- `fra` (Frankfurt, Germany)
- `syd` (Sydney, Australia)
- See all: `flyctl platform regions`

## Next Steps
After successful deploy:
1. Test the app at your Fly.io URL
2. Enable API mode in the sidebar to use external FastAPI backend (if desired)
3. Set up custom domain (optional): `flyctl certs add yourdomain.com`
