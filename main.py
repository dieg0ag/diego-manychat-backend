"""
Backend de cualificación inteligente para ManyChat.

Integraciones:
- OpenAI GPT-5.2: análisis de intención y cualificación
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

import httpx
import openai
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

MANYCHAT_TOKEN = os.environ["MANYCHAT_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.2")
NOTION_TOKEN = os.environ["NOTION_TOKEN"]
NOTION_DB_ID = os.environ.get("NOTION_DB_ID", "bfa5ea4b-cef4-4381-9f2a-a08b0f4a1d57")
CLIENTES_DS_ID = os.environ.get("CLIENTES_DS_ID", "31cc7b27-0542-81fe-a5d5-e57744aa7220")
ONBOARDING_DS_ID = os.environ.get("ONBOARDING_DS_ID", "310c7b27-0542-80e8-86a2-ca7332bed187")
ESTRATEGIAS_DS_ID = os.environ.get("ESTRATEGIAS_DS_ID", "310c7b27-0542-8096-aaa3-eed9b5349acb")
REPORTES_DS_ID = os.environ.get("REPORTES_DS_ID", "310c7b27-0542-8049-bb5a-f3f0002814ca")
BACKEND_SECRET = os.environ["BACKEND_SECRET"]

MONTHS_ES = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Ocutibre", "Noviembre", "Diciembre",
]

ONBOARDING_CHECKLIST = [
    ("FACTURA", "ADMINISTRATIVO"),
    ("DATOS DEL CLIENTE", "ADMINISTRATIVO"),
    ("FICHA NOTION", "ADMINISTRATIVO"),
    ("METRICOOL", "ADMINISTRATIVO"),
    ("DRIVE", "ADMINISTRATIVO"),
]

MANYCHAT_BASE = "https://api.manychat.com"
NOTION_BASE = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"

app = FastAPI(title="Diego Alvarez — ManyChat Qualifier")
llm = openai.OpenAI(api_key=OPENAI_API_KEY)


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


class ActivateClientRequest(BaseModel):
    subscriber_id: str | None = None
    calendly_event_id: str | None = None
    lead_page_id: str | None = None
    name: str | None = None
    email: str | None = None
    instagram: str | None = None
    phone: str | None = None
    servicio: list[str] | None = None
    precio_mensual: float | None = None
    fecha_inicio: str | None = None
    proxima_accion: str | None = None
    origen: str | None = None
    notas: str | None = None
    auto_provision: bool = True


class ProvisionRequest(BaseModel):
    client_page_id: str
    client_name: str | None = None
    mes: str | None = None


class MonthlyReportRequest(BaseModel):
    client_page_id: str
    client_name: str | None = None
    mes: str | None = None
    metricas: dict | None = None
    highlights: list[str] | None = None
    notas: str | None = None


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

    response = llm.chat.completions.create(
        model=OPENAI_MODEL,
        max_completion_tokens=400,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    analysis = _extract_json(response.choices[0].message.content)
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
    phone = invitee.get("text_reminder_number")
    questions = {q["question"]: q.get("answer") for q in invitee.get("questions_and_answers", [])}
    instagram = questions.get("Cuál es tu perfil de Instagram?") or questions.get("Cuál es tu perfil de Instagram? ")
    scheduled_event = invitee.get("scheduled_event") or {}
    event_uri = scheduled_event.get("uri") or invitee.get("uri") or ""
    calendly_event_id = event_uri.rsplit("/", 1)[-1] if event_uri else None

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
                "Telefono": questions.get("Numero de teléfono") or phone,
                "Facturacion": _normalize_facturacion(questions.get("Facturación mensual (aproximada) * ")),
                "Capacidad Inversion": _normalize_inversion(questions.get("¿Cuál es tu capacidad de inversión actual? ")),
                "Status": "sql",
                "Funnel Stage": "BOFU",
                "Llamada Agendada": True,
            },
        )

        await _notion_upsert_client(
            client,
            dedup_key=("calendly", calendly_event_id) if calendly_event_id else ("email", email),
            data={
                "Nombre": name,
                "Email": email,
                "Instagram": instagram,
                "Número de Teléfono": questions.get("Numero de teléfono") or phone,
                "Pipeline": "Auditoría agendada",
                "Origen": "Instagram ManyChat" if subscriber_id else "Directo",
                "Calendly Event ID": calendly_event_id,
                "Lead Origen ID": subscriber_id,
            },
        )
    return {"ok": True}


@app.post("/activate-client")
async def activate_client(req: ActivateClientRequest, x_secret: str = Header(None)):
    _check_secret(x_secret)

    if not any([req.calendly_event_id, req.subscriber_id, req.email, req.lead_page_id]):
        raise HTTPException(400, "need one of: calendly_event_id, subscriber_id, email, lead_page_id")

    if req.calendly_event_id:
        dedup_key = ("calendly", req.calendly_event_id)
    elif req.subscriber_id:
        dedup_key = ("subscriber", req.subscriber_id)
    elif req.email:
        dedup_key = ("email", req.email)
    else:
        dedup_key = ("page_id", req.lead_page_id)

    async with httpx.AsyncClient(timeout=15) as client:
        page = await _notion_upsert_client(
            client,
            dedup_key=dedup_key,
            data={
                "Nombre": req.name,
                "Email": req.email,
                "Instagram": req.instagram,
                "Número de Teléfono": req.phone,
                "Pipeline": "Activo",
                "Origen": req.origen or "Instagram ManyChat",
                "Servicio": req.servicio,
                "Precio Mensual": req.precio_mensual,
                "Fecha Inicio": req.fecha_inicio,
                "Próxima Acción": req.proxima_accion,
                "Lead Origen ID": req.subscriber_id,
                "Calendly Event ID": req.calendly_event_id,
                "Notas Internas": req.notas,
            },
        )

        provisioned = {}
        if req.auto_provision and page.get("id"):
            provisioned["onboarding"] = await _notion_provision_onboarding(
                client, client_page_id=page["id"], client_name=req.name
            )
            provisioned["estrategia"] = await _notion_provision_estrategia(
                client, client_page_id=page["id"], client_name=req.name
            )

        if req.subscriber_id:
            await _manychat_add_tags(client, req.subscriber_id, ["status:cliente", "pipeline:activo"])
            await _manychat_set_fields(client, req.subscriber_id, {"funnel_stage": "cliente"})

    return {"ok": True, "notion_page_id": page.get("id"), "provisioned": provisioned}


@app.post("/provision-onboarding")
async def provision_onboarding(req: ProvisionRequest, x_secret: str = Header(None)):
    _check_secret(x_secret)
    async with httpx.AsyncClient(timeout=20) as client:
        created = await _notion_provision_onboarding(
            client, client_page_id=req.client_page_id, client_name=req.client_name
        )
    return {"ok": True, "created": created}


@app.post("/provision-production")
async def provision_production(req: ProvisionRequest, x_secret: str = Header(None)):
    _check_secret(x_secret)
    async with httpx.AsyncClient(timeout=20) as client:
        created = await _notion_provision_estrategia(
            client, client_page_id=req.client_page_id, client_name=req.client_name, mes=req.mes
        )
    return {"ok": True, "created": created}


@app.post("/generate-monthly-report")
async def generate_monthly_report(req: MonthlyReportRequest, x_secret: str = Header(None)):
    _check_secret(x_secret)
    mes = req.mes or _current_month_es()
    narrative = _generate_report_narrative(req, mes) if (req.metricas or req.highlights) else None

    async with httpx.AsyncClient(timeout=30) as client:
        page = await _notion_create_report_page(client, req, mes, narrative)
    return {"ok": True, "notion_page_id": page.get("id"), "narrative": narrative}


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
        raise HTTPException(500, f"LLM no devolvió JSON: {raw[:200]}")
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


# ------------- Clientes (CRM) -------------
async def _notion_upsert_client(
    client: httpx.AsyncClient,
    dedup_key: tuple[str, str | None],
    data: dict,
) -> dict:
    existing = await _notion_find_client(client, dedup_key)
    props = _build_client_properties(data)

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
            json={"parent": {"database_id": CLIENTES_DS_ID}, "properties": props},
        )
    r.raise_for_status()
    return r.json()


async def _notion_find_client(
    client: httpx.AsyncClient,
    dedup_key: tuple[str, str | None],
) -> str | None:
    kind, value = dedup_key
    if not value:
        return None

    if kind == "page_id":
        return value

    filter_map = {
        "calendly": {"property": "Calendly Event ID", "rich_text": {"equals": value}},
        "subscriber": {"property": "Lead Origen ID", "rich_text": {"equals": value}},
        "email": {"property": "Email", "email": {"equals": value}},
    }
    if kind not in filter_map:
        return None

    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    r = await client.post(
        f"{NOTION_BASE}/databases/{CLIENTES_DS_ID}/query",
        headers=headers,
        json={"filter": filter_map[kind], "page_size": 1},
    )
    results = r.json().get("results", [])
    return results[0]["id"] if results else None


def _build_client_properties(data: dict) -> dict:
    mapping = {
        "Nombre": lambda v: {"title": [{"text": {"content": str(v)[:1900]}}]},
        "Email": lambda v: {"email": v},
        "Instagram": lambda v: {"url": v if str(v).startswith("http") else f"https://instagram.com/{str(v).lstrip('@')}"},
        "Número de Teléfono": lambda v: {"phone_number": str(v)},
        "Pipeline": lambda v: {"select": {"name": v}},
        "Origen": lambda v: {"select": {"name": v}},
        "Estado del Cliente": lambda v: {"select": {"name": v}},
        "Servicio": lambda v: {"multi_select": [{"name": s} for s in (v if isinstance(v, list) else [v])]},
        "Precio Mensual": lambda v: {"number": float(v)},
        "Fecha Inicio": lambda v: {"date": {"start": v}},
        "Próxima Acción": lambda v: {"date": {"start": v}},
        "Lead Origen ID": lambda v: {"rich_text": [{"text": {"content": str(v)[:1900]}}]},
        "Calendly Event ID": lambda v: {"rich_text": [{"text": {"content": str(v)[:1900]}}]},
        "Notas Internas": lambda v: {"rich_text": [{"text": {"content": str(v)[:1900]}}]},
        "Comentarios": lambda v: {"rich_text": [{"text": {"content": str(v)[:1900]}}]},
    }
    props: dict = {}
    for key, value in data.items():
        if value is None or value == "" or value == []:
            continue
        fn = mapping.get(key)
        if fn:
            props[key] = fn(value)
    return props


# ------------- Provisioning (Onboarding + Producción) -------------
def _current_month_es() -> str:
    return MONTHS_ES[datetime.now(timezone.utc).month - 1]


async def _notion_create_page(client: httpx.AsyncClient, parent_ds: str, properties: dict) -> dict:
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    r = await client.post(
        f"{NOTION_BASE}/pages",
        headers=headers,
        json={"parent": {"database_id": parent_ds}, "properties": properties},
    )
    r.raise_for_status()
    return r.json()


async def _notion_provision_onboarding(
    client: httpx.AsyncClient,
    client_page_id: str,
    client_name: str | None,
) -> list[str]:
    created_ids: list[str] = []
    display_name = client_name or "Cliente"
    for tag, proceso in ONBOARDING_CHECKLIST:
        props = {
            "Name": {"title": [{"text": {"content": f"{tag} — {display_name}"}}]},
            "Cliente": {"relation": [{"id": client_page_id}]},
            "Multi-select": {"multi_select": [{"name": tag}]},
            "Proceso": {"select": {"name": proceso}},
            "Status": {"status": {"name": "Omboarding"}},
            "Status 1": {"status": {"name": "Not started"}},
        }
        page = await _notion_create_page(client, ONBOARDING_DS_ID, props)
        created_ids.append(page.get("id"))
    return created_ids


async def _notion_provision_estrategia(
    client: httpx.AsyncClient,
    client_page_id: str,
    client_name: str | None,
    mes: str | None = None,
) -> list[str]:
    display_name = client_name or "Cliente"
    mes_name = mes or _current_month_es()
    props = {
        "Name": {"title": [{"text": {"content": f"Estrategia {mes_name} — {display_name}"}}]},
        "Cliente": {"relation": [{"id": client_page_id}]},
        "Proceso": {"select": {"name": "ESTRATEGIA"}},
        "Status": {"status": {"name": "Estrategias y analisis"}},
        "Mes": {"select": {"name": mes_name}},
    }
    page = await _notion_create_page(client, ESTRATEGIAS_DS_ID, props)
    return [page.get("id")]


def _generate_report_narrative(req: MonthlyReportRequest, mes: str) -> str:
    metricas_txt = json.dumps(req.metricas or {}, ensure_ascii=False, indent=2)
    highlights_txt = "\n".join(f"- {h}" for h in (req.highlights or [])) or "Ninguno"
    system = (
        "Eres el analista de contenido de Diego Alvarez. Redacta reportes mensuales para clientes de servicio DFY "
        "de edición estratégica en Instagram. Tono: directo, en tú, sin rodeos, con foco en qué funcionó, qué no, "
        "y qué haremos el próximo mes. 3 secciones: Resumen ejecutivo, Análisis, Plan próximo mes. Máximo 350 palabras."
    )
    user = (
        f"Cliente: {req.client_name or 'Cliente'}\n"
        f"Mes: {mes}\n"
        f"Métricas:\n{metricas_txt}\n"
        f"Highlights:\n{highlights_txt}\n"
        f"Notas internas: {req.notas or 'sin notas'}"
    )
    response = llm.chat.completions.create(
        model=OPENAI_MODEL,
        max_completion_tokens=800,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content.strip()


def _markdown_to_blocks(text: str) -> list[dict]:
    blocks = []
    for raw_line in text.split("\n"):
        line = raw_line.rstrip()
        if not line:
            blocks.append({"object": "block", "type": "paragraph", "paragraph": {"rich_text": []}})
            continue
        if line.startswith("### "):
            blocks.append({"object": "block", "type": "heading_3", "heading_3": {"rich_text": [{"type": "text", "text": {"content": line[4:]}}]}})
        elif line.startswith("## "):
            blocks.append({"object": "block", "type": "heading_2", "heading_2": {"rich_text": [{"type": "text", "text": {"content": line[3:]}}]}})
        elif line.startswith("# "):
            blocks.append({"object": "block", "type": "heading_1", "heading_1": {"rich_text": [{"type": "text", "text": {"content": line[2:]}}]}})
        elif line.startswith(("- ", "* ")):
            blocks.append({"object": "block", "type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": line[2:]}}]}})
        else:
            blocks.append({"object": "block", "type": "paragraph", "paragraph": {"rich_text": [{"type": "text", "text": {"content": line}}]}})
    return blocks[:95]


async def _notion_create_report_page(
    client: httpx.AsyncClient,
    req: MonthlyReportRequest,
    mes: str,
    narrative: str | None,
) -> dict:
    display_name = req.client_name or "Cliente"
    props = {
        "Name": {"title": [{"text": {"content": f"Reporte {mes} — {display_name}"}}]},
        "Cliente": {"relation": [{"id": req.client_page_id}]},
        "Proceso": {"select": {"name": "ADMINISTRATIVO"}},
        "Status": {"status": {"name": "REPORTES Y SEGUIMIENTOS"}},
        "Mes": {"select": {"name": mes}},
    }
    body = {"parent": {"database_id": REPORTES_DS_ID}, "properties": props}
    if narrative:
        body["children"] = _markdown_to_blocks(narrative)

    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    r = await client.post(f"{NOTION_BASE}/pages", headers=headers, json=body)
    r.raise_for_status()
    return r.json()
