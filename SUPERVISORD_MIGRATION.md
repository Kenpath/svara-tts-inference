# Migration to Supervisord - Summary

## Overview

Migrated from bash script-based process management to **supervisord** for production-grade deployment.

## Why Supervisord?

### Before (Bash Script)
❌ Manual process management  
❌ No automatic restart on failure  
❌ Basic signal handling  
❌ Limited logging control  
❌ Difficult to monitor individual processes  

### After (Supervisord)
✅ Automatic process restart on failure  
✅ Robust process monitoring  
✅ Built-in log rotation  
✅ Easy service control (`supervisorctl`)  
✅ Process priority and dependencies  
✅ Industry-standard solution  

## Changes Made

### 1. New Files

- **`supervisord.conf`** - Supervisord configuration
  - Manages vLLM, FastAPI, and health check processes
  - Automatic restart on failure
  - Log rotation (50MB per file, 10 backups)
  - Process priorities and dependencies

- **`scripts/supervisor_healthcheck.py`** - Health monitoring
  - Monitors both vLLM and FastAPI
  - Logs health status every 60 seconds
  - Event-driven monitoring

- **`scripts/start-dev.sh`** - Development startup script
  - Replaces old `start.sh` for local development
  - Does NOT use supervisord (for simpler dev workflow)

### 2. Modified Files

- **`Dockerfile`**
  - Added `supervisor` package
  - Copies `supervisord.conf` to `/etc/supervisor/conf.d/`
  - Changed CMD to run supervisord instead of bash script
  - Creates log directory: `/var/log/supervisor/`

- **`api/server.py`**
  - Added note that `__main__` is for development only
  - Production uses uvicorn via supervisord
  - Moved `create_speaker_id` import to module level (no local imports)

- **`DEPLOYMENT.md`**
  - Added supervisord usage section
  - Process management commands
  - Debugging with supervisorctl
  - Log viewing instructions

### 3. Deleted Files

- **`scripts/start.sh`** (moved to `start.sh.old`)
  - Replaced by supervisord for production
  - Replaced by `start-dev.sh` for development

## Supervisord Configuration

### Process Definitions

```ini
[program:vllm]
priority=100          # Starts first
autostart=true
autorestart=true
startretries=3
stopwaitsecs=30

[program:fastapi]
priority=200          # Starts after vLLM
autostart=true
autorestart=true
depends_on=vllm      # Waits for vLLM

[program:vllm-healthcheck]
priority=300          # Background monitoring
```

### Log Files

All logs in `/var/log/supervisor/`:
- `supervisord.log` - Supervisord main log
- `vllm.log` - vLLM stdout
- `vllm_error.log` - vLLM stderr
- `fastapi.log` - FastAPI stdout
- `fastapi_error.log` - FastAPI stderr
- `healthcheck.log` - Health check events

## Usage

### Production (Docker)

```bash
# Start with supervisord (automatic)
docker-compose up -d

# View supervisor status
docker-compose exec svara-tts-api supervisorctl status

# View logs
docker-compose exec svara-tts-api tail -f /var/log/supervisor/vllm.log

# Restart a service
docker-compose exec svara-tts-api supervisorctl restart vllm
docker-compose exec svara-tts-api supervisorctl restart fastapi

# Stop/start all
docker-compose exec svara-tts-api supervisorctl stop all
docker-compose exec svara-tts-api supervisorctl start all
```

### Development (Local)

```bash
# Use the development script (no supervisord)
./scripts/start-dev.sh

# OR run manually
# Terminal 1: vLLM
python -m vllm.entrypoints.openai.api_server --model kenpath/svara-tts-v1

# Terminal 2: FastAPI
cd api && python server.py
```

## Benefits

### 1. Reliability
- Automatic restart on crash
- Configurable retry attempts
- Graceful shutdown handling

### 2. Monitoring
- Real-time process status
- Health check monitoring
- Comprehensive logging

### 3. Operations
- Easy service control without container restart
- Individual service restart capability
- Log rotation (prevents disk overflow)

### 4. Scalability
- Easy to add more services
- Process dependencies (startup order)
- Priority-based startup

## Process Startup Sequence

```
1. Container starts → Supervisord initializes
   ↓
2. vLLM starts (priority 100)
   - Loads model (~30-60s)
   - Opens port 8000
   ↓
3. Health check waits for vLLM ready
   ↓
4. FastAPI starts (priority 200)
   - Connects to vLLM
   - Loads voice config
   - Opens port 8080
   ↓
5. Health monitor starts (priority 300)
   - Continuous monitoring
   ↓
6. System ready ✓
```

## Troubleshooting

### Check Process Status
```bash
docker-compose exec svara-tts-api supervisorctl status
```

### View Logs
```bash
# All logs
docker-compose exec svara-tts-api tail -f /var/log/supervisor/*.log

# Specific service
docker-compose exec svara-tts-api tail -f /var/log/supervisor/vllm.log
```

### Restart Failed Service
```bash
docker-compose exec svara-tts-api supervisorctl restart vllm
```

### Manual Control
```bash
# Stop a service
docker-compose exec svara-tts-api supervisorctl stop vllm

# Start a service
docker-compose exec svara-tts-api supervisorctl start vllm

# View detailed status
docker-compose exec svara-tts-api supervisorctl status
```

## Future Enhancements

Possible additions with supervisord:

1. **Multiple Workers**
   - Add more FastAPI workers for higher concurrency
   - Configure with `numprocs` in supervisord.conf

2. **Metrics Collection**
   - Add Prometheus exporter process
   - Monitor GPU, memory, request rates

3. **Log Forwarding**
   - Add log shipper (Fluent Bit, Filebeat)
   - Send logs to central logging system

4. **Scheduled Tasks**
   - Add cron-like tasks
   - Model cache warming
   - Health report generation

## Migration Checklist

✅ Added supervisord to Dockerfile  
✅ Created supervisord.conf  
✅ Created health check monitor  
✅ Updated Dockerfile CMD  
✅ Created development script  
✅ Moved local imports to module level  
✅ Updated documentation  
✅ Tested process management  

## Testing

To verify the migration:

```bash
# 1. Build and start
docker-compose build
docker-compose up -d

# 2. Check supervisord status
docker-compose exec svara-tts-api supervisorctl status
# Should show: vllm, fastapi, vllm-healthcheck all RUNNING

# 3. Test API
curl http://localhost:8080/health
curl http://localhost:8080/v1/voices

# 4. Test automatic restart
docker-compose exec svara-tts-api supervisorctl stop fastapi
# Wait 5 seconds
docker-compose exec svara-tts-api supervisorctl status
# Should show: fastapi back to RUNNING (auto-restarted)

# 5. View logs
docker-compose exec svara-tts-api tail -20 /var/log/supervisor/healthcheck.log
```

## Conclusion

The migration to supervisord provides a production-ready deployment with:
- Better reliability through automatic restarts
- Improved monitoring and logging
- Easier operations and debugging
- Foundation for future scaling

This is now the **recommended deployment method** for production use.

