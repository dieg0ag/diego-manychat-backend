"""
Backend de cualificación inteligente para ManyChat.

Integraciones:
- Claude Haiku: análisis de intención y cualificación
- Notion: crea/actualiza leads en base "Leads ManyChat"
- Calendly: detecta cuándo un lead ha agendado (webhook entrante)
- ManyChat: sincroniza custom fields y tags

Endpoints:
- POST /qualify           → ManyChat manda respuesta libre, devolvemos análisis + acciones
- POST /sync-lead         → ManyChat manda evento importante (después de filtro, entrega LM, etc.)
- POST /calendly-webhook  → Calendly notifica cuando alguien agenda (Zapier o webhook directo)
- GET  /health            → healthcheck
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Literal

import anthropic
import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

MANYCHAT_TOKEN = os.environ["MANYCHAT_TOKEN"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
NOTION_TOKEN = os.environ["NOTION_TOKEN"]
NOTION_DB_ID = os.environ.get("NOTION_DB_ID", "e51d9d2b-c58f-428c-b0ab-9804624e24dd")
BACKEND_SECRET = os.environ["BACKEND_SECRET"]

MANYCHAT_BASE = "https://api.manychat.com"
NOTION_BASE = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"

app = FastAPI(title="Diego Alvarez — ManyChat Qualifier")
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# ------------- Models -------------
class QualifyRequest(BaseModel):
    subscriber_id: str
    message: str
    niche: str | None = None
    goal: str | None = None
    lead_score: int = 0
    first_name: str | None = None


class SyncLeadRequest(BaseModel):
    subscriber_id: str
    first_name: str | None = None
    last_name: str | None = None
    instagram: str | None = None
    email: str | None = None
    phone: str | None = None
    status: str | None = None
    funnel_stage: str | None = None
    lead_score: int | None = None
    calificacion: str | None = None
    niche: str | None = None
    followers_range: str | None = None
    goal: str | None = None
    objection: str | None = None
    lead_magnet: str | None = None
    source: str | None = None
    followup_count: int | None = None
    notes: str | None = None


class CalendlyWebhook(BaseModel):
    event: str
    payload: dict


class ManyChatV2Response(BaseModel):
    version: Literal["v2"] = "v2"
    content: dict


SYSTEM_PROMPT = """Eres el asistente de cualificación de Diego Alvarez, editor y estratega que delega la edición de contenido a coaches, consultores, terapeutas, agencias y creadores que ya facturan.

Producto principal: servicio done-for-you de edición estratégica de contenido en Instagram.
ICP ideal: 2k-30k€/mes facturación, capacidad inversión 2-4k€+.
Llamada BOFU: Auditoría 1:1 de 60 min (https://calendly.com/diegoagtrabajo/reuniones).

Tu trabajo: analizar el mensaje libre del lead y devolver SOLO JSON estricto.

{
  "score_delta": int entre -10 y +40,
  "objection": "precio"|"tiempo"|"confianza"|"no_urgencia"|"ninguna",
  "funnel_stage": "TOFU"|"MOFU"|"BOFU",
  "next_action": "agendar_llamada"|"enviar_clase"|"nurturing"|"descartar",
  "personal_reply": string corto 1-2 frases en TÚ, directo, sin emojis excesivos, tono Diego Alvarez
}

Reglas de score:
- Menciona dinero/presupuesto/ROI → BOFU, +30
- Ya vende y pide ayuda concreta → MOFU/BOFU, +25
- Curiosidad general → TOFU, 0
- Empieza desde cero sin negocio → TOFU, -5

Reglas de objeción:
- "caro/precio/no puedo pagar" → precio
- "sin tiempo/ocupado/más adelante" → tiempo
- "¿funciona?/resultados/garantía" → confianza
- "luego/algún día/no es momento" → no_urgencia
"""


# ------------- Endpoints -------------
@app.post("/qualify", response_model=ManyChatV2Response)
async def qualify(req: QualifyRequest, x_secret: str = Header(None)):
    _check_secret(x_secret)

    context = (
        f"Nicho: {req.niche or 'desconocido'}. "
        f"Meta: {req.goal or 'desconocida'}. "
        f"Score actual: {req.lead_score}. "
        f"Nombre: {req.first_name or 'sin nombre'}."
    )
    user_msg = f"{context}\n\nMensaje del lead:\n{req.message}"

    response = claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    analysis = _extract_json(response.content[0].text)
    new_score = max(0, min(100, req.lead_score + int(analysis["score_delta"])))

    async with httpx.AsyncClient(timeout=15) as client:
        await _manychat_set_fields(
            client,
            req.subscriber_id,
            {
                "lead_score": new_score,
                "objection_detected": analysis["objection"],
                "funnel_stage": analysis["funnel_stage"],
            },
        )
        await _notion_upsert_lead(
            client,
            subscriber_id=req.subscriber_id,
            data={
                "Lead Score": new_score,
                "Objecion": analysis["objection"],
                "Funnel Stage": analysis["funnel_stage"],
                "Notas": req.message[:1900],
            },
        )

    return ManyChatV2Response(
        content={
            "messages": [{"type": "text", "text": analysis["personal_reply"]}],
            "actions": [
                {"action": "set_field_value", "field_name": "lead_score", "value": new_score},
                {"action": "set_field_value", "field_name": "objection_detected", "value": analysis["objection"]},
                {"action": "set_field_value", "field_name": "funnel_stage", "value": analysis["funnel_stage"]},
                {"action": "add_tag", "tag_name": _action_to_tag(analysis["next_action"])},
            ],
        }
    )


@app.post("/sync-lead")
async def sync_lead(req: SyncLeadRequest, x_secret: str = Header(None)):
    _check_secret(x_secret)

    notion_data = _req_to_notion_props(req)
    async with httpx.AsyncClient(timeout=15) as client:
        page = await _notion_upsert_lead(client, subscriber_id=req.subscriber_id, data=notion_data)
    return {"ok": True, "notion_page_id": page.get("id")}


@app.post("/calendly-webhook")
async def calendly_webhook(body: CalendlyWebhook):
    if body.event != "invitee.created":
        return {"ignored": True}

    invitee = body.payload
    email = invitee.get("email")
    name = invitee.get("name")
    questions = {q["question"]: q.get("answer") for q in invitee.get("questions_and_answers", [])}
    instagram = questions.get("Cuál es tu perfil de Instagram?") or questions.get("Cuál es tu perfil de Instagram? ")

    async with httpx.AsyncClient(timeout=15) as client:
        subscriber_id = await _find_manychat_subscriber_by_instagram(client, instagram) if instagram else None

        if subscriber_id:
            await _manychat_add_tags(client, subscriber_id, ["status:sql", "accion:agendar_llamada"])
            await _manychat_set_fields(client, subscriber_id, {"funnel_stage": "BOFU"})

        await _notion_upsert_lead(
            client,
            subscriber_id=subscriber_id or f"calendly:{email}",
            data={
                "Nombre": name,
                "Email": email,
                "Instagram": instagram,
                "Telefono": questions.get("Numero de teléfono"),
                "Facturacion": _normalize_facturacion(questions.get("Facturación mensual (aproximada) * ")),
                "Capacidad Inversion": _normalize_inversion(questions.get("¿Cuál es tu capacidad de inversión actual? ")),
                "Status": "sql",
                "Funnel Stage": "BOFU",
                "Llamada Agendada": True,
            },
        )
    return {"ok": True}


@app.get("/health")
async def health():
    return {"ok": True, "ts": datetime.now(timezone.utc).isoformat()}


# ------------- Helpers -------------
def _check_secret(x_secret: str | None):
    if x_secret != BACKEND_SECRET:
        raise HTTPException(401, "unauthorized")


def _extract_json(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise HTTPException(500, f"Claude no devolvió JSON: {raw[:200]}")
    return json.loads(match.group(0))


def _action_to_tag(action: str) -> str:
    return {
        "agendar_llamada": "accion:agendar_llamada",
        "enviar_clase": "accion:enviar_recurso",
        "nurturing": "accion:seguir_nurturing",
        "descartar": "status:no_interesado",
    }.get(action, "accion:seguir_nurturing")


async def _manychat_set_fields(client: httpx.AsyncClient, subscriber_id: str, fields: dict):
    headers = {"Authorization": f"Bearer {MANYCHAT_TOKEN}"}
    for name, value in fields.items():
        if value is None:
            continue
        await client.post(
            f"{MANYCHAT_BASE}/fb/subscriber/setCustomFieldByName",
            headers=headers,
            json={"subscriber_id": subscriber_id, "field_name": name, "field_value": value},
        )


async def _manychat_add_tags(client: httpx.AsyncClient, subscriber_id: str, tags: list[str]):
    headers = {"Authorization": f"Bearer {MANYCHAT_TOKEN}"}
    for tag in tags:
        await client.post(
            f"{MANYCHAT_BASE}/fb/subscriber/addTagByName",
            headers=headers,
            json={"subscriber_id": subscriber_id, "tag_name": tag},
        )


async def _find_manychat_subscriber_by_instagram(client: httpx.AsyncClient, ig_handle: str) -> str | None:
    headers = {"Authorization": f"Bearer {MANYCHAT_TOKEN}"}
    r = await client.get(
        f"{MANYCHAT_BASE}/fb/subscriber/findByCustomField",
        headers=headers,
        params={"field_id": 14507674, "field_value": ig_handle.replace("@", "")},
    )
    data = r.json().get("data") or []
    return data[0]["id"] if data else None


async def _notion_upsert_lead(client: httpx.AsyncClient, subscriber_id: str, data: dict) -> dict:
    existing = await _notion_find_by_subscriber(client, subscriber_id)
    props = _build_notion_properties(data, subscriber_id)

    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }

    if existing:
        r = await client.patch(
            f"{NOTION_BASE}/pages/{existing}",
            headers=headers,
            json={"properties": props},
        )
    else:
        r = await client.post(
            f"{NOTION_BASE}/pages",
            headers=headers,
            json={"parent": {"database_id": NOTION_DB_ID}, "properties": props},
        )
    r.raise_for_status()
    return r.json()


async def _notion_find_by_subscriber(client: httpx.AsyncClient, subscriber_id: str) -> str | None:
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    r = await client.post(
        f"{NOTION_BASE}/databases/{NOTION_DB_ID}/query",
        headers=headers,
        json={
            "filter": {"property": "Subscriber ID", "rich_text": {"equals": subscriber_id}},
            "page_size": 1,
        },
    )
    results = r.json().get("results", [])
    return results[0]["id"] if results else None


def _build_notion_properties(data: dict, subscriber_id: str) -> dict:
    props: dict = {"Subscriber ID": {"rich_text": [{"text": {"content": subscriber_id}}]}}
    mapping = {
        "Nombre": lambda v: {"title": [{"text": {"content": v}}]},
        "Email": lambda v: {"email": v},
        "Telefono": lambda v: {"phone_number": v},
        "Instagram": lambda v: {"url": v if v.startswith("http") else f"https://instagram.com/{v.lstrip('@')}"},
        "Status": lambda v: {"select": {"name": v}},
        "Funnel Stage": lambda v: {"select": {"name": v}},
        "Calificacion": lambda v: {"select": {"name": v}},
        "Nicho": lambda v: {"select": {"name": v}},
        "Seguidores": lambda v: {"select": {"name": v}},
        "Facturacion": lambda v: {"select": {"name": v}},
        "Capacidad Inversion": lambda v: {"select": {"name": v}},
        "Objecion": lambda v: {"select": {"name": v}},
        "Fuente": lambda v: {"select": {"name": v}},
        "Lead Score": lambda v: {"number": int(v)},
        "Followup Count": lambda v: {"number": int(v)},
        "Llamada Agendada": lambda v: {"checkbox": bool(v)},
        "Lead Magnet": lambda v: {"rich_text": [{"text": {"content": str(v)[:1900]}}]},
        "Meta": lambda v: {"rich_text": [{"text": {"content": str(v)[:1900]}}]},
        "Notas": lambda v: {"rich_text": [{"text": {"content": str(v)[:1900]}}]},
    }
    for key, value in data.items():
        if value is None or value == "":
            continue
        fn = mapping.get(key)
        if fn:
            props[key] = fn(value)
    return props


def _req_to_notion_props(req: SyncLeadRequest) -> dict:
    return {
        "Nombre": f"{req.first_name or ''} {req.last_name or ''}".strip() or None,
        "Instagram": req.instagram,
        "Email": req.email,
        "Telefono": req.phone,
        "Status": req.status,
        "Funnel Stage": req.funnel_stage,
        "Lead Score": req.lead_score,
        "Calificacion": req.calificacion,
        "Nicho": req.niche,
        "Seguidores": req.followers_range,
        "Meta": req.goal,
        "Objecion": req.objection,
        "Lead Magnet": req.lead_magnet,
        "Fuente": req.source,
        "Followup Count": req.followup_count,
        "Notas": req.notes,
    }


def _normalize_facturacion(raw: str | None) -> str | None:
    if not raw:
        return None
    m = {
        "Menos de 1000€/": "<1k",
        "Entre 1.000€ y 2.000€/mes": "1-2k",
        "Entre 2.000€ y 5.000€/mes": "2-5k",
        "Entre 5.000€ y 10.000€/mes": "5-10k",
        "Entre 10.000€ y 30.000€/mes": "10-30k",
        "+30.000€/mes": "30k+",
    }
    return m.get(raw)


def _normalize_inversion(raw: str | None) -> str | None:
    if not raw:
        return None
    m = {"1000€ o menos": "<1k", "2000€ - 4000€": "2-4k", "+5000€": "5k+"}
    return m.get(raw)
