#!/usr/bin/env bash
# Deploy completo del backend a Railway + configuración ManyChat
# Uso: ./deploy.sh
# Requiere: haber hecho `railway login` antes, y tener los 3 tokens en variables.

set -e

cd "$(dirname "$0")"

: "${MANYCHAT_TOKEN:?falta MANYCHAT_TOKEN}"
: "${OPENAI_API_KEY:?falta OPENAI_API_KEY}"
: "${NOTION_TOKEN:?falta NOTION_TOKEN}"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.2}"

BACKEND_SECRET="${BACKEND_SECRET:-$(openssl rand -hex 32)}"
echo "BACKEND_SECRET=$BACKEND_SECRET"

echo "→ Inicializando proyecto Railway..."
railway init --name manychat-qualifier 2>/dev/null || echo "(ya existe, reutilizando)"

echo "→ Seteando variables..."
railway variables \
  --set MANYCHAT_TOKEN="$MANYCHAT_TOKEN" \
  --set OPENAI_API_KEY="$OPENAI_API_KEY" \
  --set OPENAI_MODEL="$OPENAI_MODEL" \
  --set NOTION_TOKEN="$NOTION_TOKEN" \
  --set NOTION_DB_ID="bfa5ea4b-cef4-4381-9f2a-a08b0f4a1d57" \
  --set CLIENTES_DS_ID="31cc7b27-0542-81fe-a5d5-e57744aa7220" \
  --set ONBOARDING_DS_ID="310c7b27-0542-80e8-86a2-ca7332bed187" \
  --set ESTRATEGIAS_DS_ID="310c7b27-0542-8096-aaa3-eed9b5349acb" \
  --set REPORTES_DS_ID="310c7b27-0542-8049-bb5a-f3f0002814ca" \
  --set BACKEND_SECRET="$BACKEND_SECRET"

echo "→ Deploy..."
railway up --detach

echo "→ Generando dominio público..."
DOMAIN=$(railway domain 2>&1 | grep -oE 'https://[^ ]+' | head -1 || true)
if [ -z "$DOMAIN" ]; then
  DOMAIN=$(railway domain generate 2>&1 | grep -oE 'https://[^ ]+' | head -1 || true)
fi

echo ""
echo "=== RESULTADO ==="
echo "Dominio backend: $DOMAIN"
echo "BACKEND_SECRET (guárdalo): $BACKEND_SECRET"
echo ""

if [ -n "$DOMAIN" ]; then
  echo "→ Verificando /health..."
  sleep 5
  curl -sS "$DOMAIN/health" || echo "(aún arrancando, prueba en 30s)"
  echo ""

  echo "→ Guardando URL en ManyChat bot field webhook_backend..."
  curl -sS -X POST "https://api.manychat.com/fb/page/setBotField" \
    -H "Authorization: Bearer $MANYCHAT_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"field_id\":5011438,\"field_value\":\"$DOMAIN\"}" | head -c 200
  echo ""
fi

echo ""
echo "Guarda estos valores:"
echo "  BACKEND_URL=$DOMAIN"
echo "  BACKEND_SECRET=$BACKEND_SECRET"
