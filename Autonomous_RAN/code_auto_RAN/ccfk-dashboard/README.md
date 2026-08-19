# Build CCFK React Dashboard (requires Node.js >= 18)

## Prerequisites
- Node.js 18+ and npm 9+
- Cloned CCFK repo at `../../ccfk` (already done)
- Nokia `.npmrc` registry access (see `ccfk/.npmrc`)

## Quick build (PowerShell)

```powershell
# 1. Build CCFK library
Set-Location ..\..\ccfk
npm install --legacy-peer-deps
npm run build-ts

# 2. Build Autonomous RAN CCFK dashboard
Set-Location ..\code_auto_RAN\ccfk-dashboard
npm install --legacy-peer-deps
npm run build

# 3. Restart API server
Set-Location ..
python scripts/run_api_server.py
```

## URLs
| Dashboard | URL |
|-----------|-----|
| Hub (choose view) | http://localhost:8080/dashboard |
| Classic | http://localhost:8080/dashboard/classic |
| CCFK (static + theme) | http://localhost:8080/dashboard/ccfk |
| CCFK React (after build) | Same URL — replaces static `index.html` in `static/ccfk-dashboard/` |

## Development
```powershell
cd code_auto_RAN\ccfk-dashboard
npm run dev   # Vite dev server on :5173, proxies /api to :8080
```
