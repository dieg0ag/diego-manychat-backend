# Backend de cualificación inteligente

Backend FastAPI que cualifica leads con Claude Haiku cuando ManyChat recibe respuestas en texto libre.

## Setup local

```bash
cd backend
uv sync
cp .env.example .env
# edita .env con tus claves reales
uv run uvicorn main:app --reload --port 8000
```

## Deploy (opciones)

**Railway** (más fácil):
```bash
railway login
railway init
railway up
```

**Fly.io:**
```bash
fly launch
fly secrets set MANYCHAT_TOKEN=... ANTHROPIC_API_KEY=... BACKEND_SECRET=...
fly deploy
```

**Render:** conecta el repo y define las 3 env vars.

Una vez deployado, guarda la URL pública en ManyChat → Settings → Bot Fields → `webhook_backend`.

## Configurar en ManyChat

Dentro del Flow `01 — FILTRO Cualificación`, después de las quick replies o cuando el lead responde en texto libre:

1. Añadir nodo **External Request** (requiere Pro)
2. Config:
   - URL: `{{webhook_backend}}/qualify`
   - Method: POST
   - Headers:
     - `Content-Type: application/json`
     - `X-Secret: {el BACKEND_SECRET que definiste}`
   - Body:
     ```json
     {
       "subscriber_id": "{{contact.id}}",
       "message": "{{last_input_text}}",
       "niche": "{{niche}}",
       "goal": "{{goal}}",
       "lead_score": {{lead_score}}
     }
     ```

3. **Response type**: "ManyChat response" (v2)
   - El backend ya devuelve `{version: "v2", content: {...}}` con messages y actions, ManyChat los aplica solo.

4. Después del External Request, añade una Condition sobre `accion:*` tag para enrutar a HOT/WARM/COLD.

## Coste estimado

- Claude Haiku ~$0.0003 por lead cualificado
- 1000 leads/mes = $0.30
- Despreciable.

## Seguridad

- El header `X-Secret` protege tu endpoint.
- La API key de ManyChat queda solo server-side, nunca expuesta al cliente.
- Genera el secret con: `openssl rand -hex 32`
