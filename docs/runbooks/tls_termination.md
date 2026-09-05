# TLS Termination Runbook (P0-2)

NeuralGuard must not receive prompts over plain HTTP in production. The
application lifespan refuses to start in `production` unless
`NEURALGUARD_ALLOW_INSECURE_HTTP=true` is set — and that flag must
ONLY be set when a TLS-terminating reverse proxy is in front.

## Option A — TLS at a reverse proxy (recommended)

Terminate TLS at nginx/Caddy/Traefik and proxy to NeuralGuard over a
loopback socket. Prompts never leave the host unencrypted.

### Caddy (one file)

```caddyfile
neuralguard.example.com {
    reverse_proxy 127.0.0.1:8000
    # Caddy manages certs automatically (Let's Encrypt / ZeroSSL).
    # Optional: client-IP allowlisting, request body limits.
}
```

### nginx

```nginx
server {
    listen 443 ssl http2;
    server_name neuralguard.example.com;

    ssl_certificate     /etc/ssl/neuralguard/fullchain.pem;
    ssl_certificate_key /etc/ssl/neuralguard/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;

    # NeuralGuard already caps request bodies; mirror the limit here.
    client_max_body_size 1m;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
    }
}

# Redirect plaintext to TLS.
server {
    listen 80;
    server_name neuralguard.example.com;
    return 301 https://$host$request_uri;
}
```

### NeuralGuard env (behind the proxy)

```bash
NEURALGUARD_ENVIRONMENT=production
NEURALGUARD_ALLOW_INSECURE_HTTP=true   # OK: TLS terminated upstream
NEURALGUARD_HOST=127.0.0.1             # bind loopback only
NEURALGUARD_PORT=8000
```

Verify: `curl -sk https://neuralguard.example.com/v1/health` returns 200;
`curl http://neuralguard.example.com/v1/health` returns a 301 (or is
refused). NeuralGuard's own `production_tls_notice` log line is expected.

## Option B — TLS at uvicorn (no proxy)

For single-host deploys without a proxy, terminate TLS in uvicorn directly.
Set `allow_insecure_http=false` (the default) and pass cert/key to uvicorn.

```bash
uv run uvicorn neuralguard.main:create_app --factory \
  --host 0.0.0.0 --port 8443 \
  --ssl-keyfile /etc/ssl/neuralguard/privkey.pem \
  --ssl-certfile /etc/ssl/neuralguard/fullchain.pem
```

```bash
NEURALGUARD_ENVIRONMENT=production
NEURALGUARD_ALLOW_INSECURE_HTTP=false
```

Verify: `curl -sk https://host:8443/v1/health` → 200; plain HTTP on 8443
fails the handshake.

## Cert renewal

- Caddy: automatic, no cron needed.
- nginx/certbot: `certbot renew --deploy-hook "systemctl reload nginx"`
  in a weekly cron.
- uvicorn: restart the service after cert rotation (no hot-reload of
  ssl-certfile). Schedule a rolling restart.

## What is NOT covered here

- mTLS between NeuralGuard and Ollama (the judge). Keep `judge_ollama_url`
  on `http://127.0.0.1:11434` so prompts stay on the loopback. If Ollama is
  remote, tunnel over WireGuard/SSH — do not expose it on the network.
- HSTS: add `Strict-Transport-Security` at the proxy for browser-facing
  deployments. NeuralGuard is API-only by default (no CORS unless configured).