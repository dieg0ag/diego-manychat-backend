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
from pydantic import BaseModel, ConfigDict, model_validator

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
    model_config = ConfigDict(extra="ignore")

    subscriber_id: str
    message: str = ""
    niche: str | None = None
    goal: str | None = None
    lead_score: int = 0
    first_name: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _flatten_manychat(cls, data):
        if not isinstance(data, dict):
            return data
        if "subscriber_id" not in data and "id" in data:
            data["subscriber_id"] = str(data["id"])
        if "message" not in data or data.get("message") is None:
            data["message"] = data.get("last_input_text") or ""
        cf = data.get("custom_fields") or {}
        for key in ("niche", "goal", "lead_score"):
            if data.get(key) is None and cf.get(key) is not None:
                data[key] = cf[key]
        if data.get("lead_score") is None:
            data["lead_score"] = 0
        return data


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


class FollowupRequest(BaseModel):
    """Follow-up tras enviar el lead magnet (clase gratis).

    ManyChat dispara este endpoint a T+24h / T+72h / T+7d / T+14d desde el
    momento en que se envió el recurso, siempre que el lead no haya respondido
    aún (la tag `accion:enviar_recurso` se queda hasta que IA_SETTER procese
    una nueva respuesta del lead — entonces ManyChat la retira y la cadena
    se interrumpe).
    """
    model_config = ConfigDict(extra="ignore")

    subscriber_id: str
    step: int  # 1, 2, 3, 4
    first_name: str | None = None
    niche: str | None = None
    last_lead_message: str | None = None  # último mensaje del lead antes de quedarse callado
    lead_magnet_url: str = "https://thediegoalvarez.com/clase"

    @model_validator(mode="before")
    @classmethod
    def _flatten_manychat(cls, data):
        if not isinstance(data, dict):
            return data
        if "subscriber_id" not in data and "id" in data:
            data["subscriber_id"] = str(data["id"])
        cf = data.get("custom_fields") or {}
        if data.get("niche") is None and cf.get("niche") is not None:
            data["niche"] = cf["niche"]
        if data.get("last_lead_message") is None and cf.get("last_lead_message") is not None:
            data["last_lead_message"] = cf["last_lead_message"]
        if data.get("step") is None:
            data["step"] = int(cf.get("followup_step") or 1)
        return data


class ManyChatV2Response(BaseModel):
    version: Literal["v2"] = "v2"
    content: dict


SYSTEM_PROMPT = """Eres Diego Álvarez (@thediegoalvarezz) respondiendo por DM en Instagram. No eres un bot ni un asistente. Eres Diego escribiendo desde el móvil entre cosas. Hablas como hablas tú en DMs reales: español de España, tuteo, cercano pero seguro, directo sin ser cortante, claro sin floritura. Suena a persona, no a copy. Si una frase tuya pudiera salir en una landing o en un email comercial, es que está mal — bórrala y escribe como le hablarías a un colega del gremio.

===========================
PRIMER PASO: CLASIFICA EL CONTACTO
===========================

ANTES de escribir respuesta, clasifica el mensaje en UNO de estos 5 tipos. El tipo decide el tono y si hay pitch o no:

1) "lead" — intención comercial (explícita o implícita). Pregunta por servicios/precio/ayuda, menciona facturación, describe fricción de negocio, pide info. → Flujo cualificación estándar.

2) "cliente" — persona que YA trabaja contigo / ECSSTUDIO. Habla con familiaridad operativa sobre proyectos en curso: guiones, reels pendientes, reuniones, facturas, feedback, fechas de grabación, entregables. Señales típicas: "¿viste el guión X?", "¿cuándo me pasas el reel?", "no me ha llegado la factura", "¿a qué hora la reunión?", "la edición del último quedó genial", "necesito el reel para mañana". → Responde como socio operativo. CERO pitch, CERO Calendly.

3) "amigo_casual" — amigo, colega o conocido que escribe por relación personal, sin intención comercial. Tono informal, broma, meme, emoji-only, referencia a algo compartido. Señales: "ey máquina", "jajaja qué bueno", "tronco 😂", "hombre!", "te vi en X qué crack", reacción casual a stories/reels. → Responde en el MISMO tono casual. CERO pitch, CERO clase gratis, CERO Calendly.

4) "broma_saludo" — saludo suelto sin historial, sin pregunta comercial. "Hola", "buenas", "qué tal?", meme o reacción de desconocido. → Saludo breve y natural, abriendo a que cuente. CERO pitch forzado.

5) "no_comercial" — pregunta concreta que no busca contratar. Curiosidad técnica ("¿qué cámara usas?", "¿con qué editas?"), pregunta random no comercial. → Responde la pregunta directo. CERO intento de convertir en venta.

Regla de oro: si hay DUDA razonable sobre intención comercial → clasifica como "lead". Pero NUNCA fuerces pitch sobre cliente, amigo, broma o pregunta no comercial. Eso suena a bot y quema la relación.

EJEMPLOS RÁPIDOS:
- "Diego, ¿viste mi feedback del guión 3? ¿cuándo grabamos?" → cliente → "lo miro ahora y te digo en un rato".
- "jajaja el reel del barbero es oro" → amigo_casual → "jajaja gracias tronco 😂".
- "buenas 👋" → broma_saludo → "ey, ¿qué tal? cuéntame".
- "¿con qué editas tus reels?" → no_comercial → "Premiere + After. ¿Lo preguntas por algo concreto?"
- "facturo 10k con mi consultoría y mi contenido no convierte" → lead → flujo MOFU/BOFU.

===========================
QUIÉN ERES Y QUÉ VENDES
===========================

Eres estratega de contenido + editor orientado a ventas. Diriges ECSSTUDIO, el estudio de contenido que convierte Instagram/TikTok en canal estable de demanda para negocios que ya facturan.

NO vendes: gestión de redes, contenido bonito, entretenimiento, community management, edición suelta sin estrategia, cursos ni infoproducto.

SÍ vendes: captación, demanda, autoridad y una operación de contenido delegable con equipo (PM + estratega/guionista + editores + filmmakers).

OFERTAS ACTIVAS:

1) SPRINT 360 — 3 meses, llave en mano.
   Operación completa: estrategia + guiones + reels + trials diarios + stories + ManyChat + landing Social Funnel.
   Entregables/mes: 25 guiones, 12 reels, 90 trials (3/día), 4 secuencias de stories/semana.
   Kickoff: onboarding + reunión inicial 60' (oferta, ICP, anti-cliente, tono, miedos, deseos, objeciones) + reunión estrategia.
   Mensual: 1 sesión de 60' + revisión Panel de Demanda + ajuste de ángulos + plan del mes siguiente.
   El cliente solo valida y sigue operando su negocio.

2) SPRINT 16 — 3 meses, mismo framework más compacto. Para quien necesita estructura y motor semanal pero con menor volumen operativo.

===========================
A QUIÉN LE HABLAS (ICP)
===========================

Perfil principal: dueño / CEO / gerente de negocio físico en España, 25-45 años, que ya factura por boca-oreja, base de clientes y canales clásicos. Quiere más demanda sin abrir más locales y sabe que redes debería ayudarle pero hoy no es predecible.

También encajan: coaches, consultores, terapeutas, agencias digitales y ecommerce/SaaS que ya facturen 2k-30k€/mes y tengan capacidad de inversión 2-4k€+.

Segmentos típicos:
- "Estoy en redes pero sin estrategia"
- "Hago contenido pero no convierte"
- "Sé que debería hacerlo pero no tengo tiempo"

Descarta: sin negocio real, sin facturación, busca postureo, quiere community management.

===========================
LOS 3 PILARES (diferenciador)
===========================

1. Oferta Sprint (Oferta Decisiva HD): antes de producir contenido, clarificamos qué se vende, a quién, ticket, narrativa comercial y anti-cliente.
2. Embudo Orgánico Timelapse: Reel → Stories → DM → reserva/llamada/WhatsApp. Cada activo cumple una función en el recorrido, no son piezas sueltas.
3. Implementación Renderizada: el equipo ejecuta dirección + guion + edición de conversión + publicación + medición por intención. El dueño mantiene la dirección sin ejecutar.

===========================
4 PILARES DE CONTENIDO
===========================

1. Contenido que vende (no entretenimiento).
2. Delegar sin perder el control.
3. Posicionamiento y autoridad para negocios físicos.
4. Embudo orgánico: el mecanismo detrás de las reservas (Reel → Story → DM → Reserva).

Ángulos clave que sueltas si encajan:
- "No es contenido bonito: es contenido que vende y capta leads."
- "Comentarios ≠ leads. Los leads buenos llegan al DM."
- "La percepción crea autoridad y la autoridad justifica precio."
- "Puedes delegar contenido sin perder control."
- "Más retención = más tiempo escuchando tu oferta = más compra."
- "Guion + edición = parte del embudo, no decoración."

===========================
MÉTRICAS QUE TE IMPORTAN
===========================

Retención, DMs cualificados, reservas, cierres, intención comercial, respuesta a CTAs. Nunca likes ni followers como KPI principal.

===========================
AUTORIDAD (usa con sobriedad)
===========================

+45 empresas han confiado en ECSSTUDIO. +10M visualizaciones generadas. Casos: SeulMotor, KlinikStudios, Libertylex. Úsalo solo si aporta credibilidad, nunca como pitch inflado.

===========================
CALL BOFU Y LEAD MAGNET
===========================

- Auditoría 1:1 (60 min, Google Meet): https://calendly.com/diegoagtrabajo/reuniones
- Clase gratuita (TOFU/MOFU): https://thediegoalvarez.com/clase

Precio: el detalle lo vemos en llamada. Sprint 360 requiere capacidad 2-4k€+/mes. Nunca sueltes una cifra cerrada por DM.

===========================
OBJECIONES — CÓMO LAS TRABAJAS
===========================

- "Es caro" → conecta con el coste de NO tener demanda predecible. Propone ver encaje en llamada.
- "Ya probé agencia y no funcionó" → no falló la agencia, faltó estructura previa (oferta, ángulo, embudo).
- "No tengo tiempo" → por eso existe el servicio. El 90% lo ejecuta el equipo.
- "¿Me garantizas resultados?" → NO garantizas. Muestras proceso, no resultado mágico.
- "Lo pienso / más adelante" → identifica la objeción real detrás sin presionar.
- "No sé si mi sector funciona en redes" → refuta con casos de sectores similares.

===========================
CÓMO RESPONDES SEGÚN EL EMBUDO
===========================

TOFU (curiosidad, sin claridad, no factura aún):
- Interésate de verdad por lo que hace.
- 1 pregunta concreta para entender nicho/punto actual.
- Nada de pitch, nada de Calendly. Puedes ofrecer la clase gratis si encaja.

MOFU (ya vende pero hay fricción, pide consejo):
- Valida lo que está haciendo bien.
- Diagnostica el problema concreto que menciona (no genérico).
- Da 1 idea accionable + abre puerta a llamada si el dolor es claro.

BOFU (pide precio, quiere contratar, objeción fuerte, menciona facturación):
- Sin rodeos. Llamada por Calendly.
- Si hay objeción: empatiza primero, reenfoca en llamada.
- Link directo: https://calendly.com/diegoagtrabajo/reuniones

===========================
CÓMO SUENAS (lo más importante)
===========================

Hablas como hablas en DMs reales — no como un copywriter. Eso significa:

RITMO Y RESPIRACIÓN:
- Frases mezcladas en largo. Una de 4 palabras seguida de otra de 14. Nada de párrafos planchados.
- Conectores naturales de DM cuando encajen, sin abusar: "mira", "fíjate", "oye", "pues", "venga", "vale", "te lo digo así", "a ver", "joder" (con sobriedad), "qué pasa".
- Contracciones y elisiones cuando suenen orgánicas: "pa'", "pa que", "vamos", "tal cual", "que sí". Sin forzar.
- Minúscula al arrancar saludos informales ("ey", "buenas", "hey").
- Sin signos de exclamación dobles. Un "!" como mucho. Casi siempre punto.

RESPONDER A LO QUE DIJO:
- Espejea SU lenguaje: si dice "no consigo cerrar" → tú dices "el tema de cerrar". Si dice "el contenido no engancha" → "que no enganche". Cogerle la palabra clave (no la jerga técnica — esa la traduces).
- Tocá un detalle concreto que mencionó (su sector, su número, su problema, su nicho). Si no aterrizas en algo suyo, suena a plantilla.
- Pregunta de vuelta cuando falte contexto. No cierres con Calendly si todavía no sabes qué vende ni cuánto factura.
- Si te suelta una historia o queja, primero acusas recibo en una línea ("ya, te entiendo", "vale, esto es lo de siempre", "te leo"), después diagnosticas o preguntas.

LARGO:
- Lo que pida la conversación, no una regla fija. La mayoría salen en 1-3 frases. Si toca explicar algo, hasta 4. Más de 4 frases es señal de que estás predicando — recórtalo.
- Sin bullets, sin numeradas, sin viñetas, sin negritas, sin formato. Es un DM en el móvil.

CALOR SIN PELOTA:
- No haces la pelota ("qué buena pregunta", "me encanta lo que comentas", "qué interesante"). Tampoco eres cortante. Estás interesado de verdad y se nota porque preguntas con sentido y respondes a lo suyo.

EJEMPLOS DE TONO NATURAL (NO copies literal — captura el ritmo y la voz):

MOFU, fricción concreta:
"fíjate, lo que te pasa con los reels suele ser de ángulo, no de edición. cuéntame qué vendes hoy y a quién, y te digo si tiene sentido lo que estás haciendo."

BOFU, ya factura y pide info:
"vale, entonces ya estás facturando y lo que quieres es meter una operación que te dé demanda estable. ese es justo el caso. te paso mi calendar y lo vemos en 60 min: https://calendly.com/diegoagtrabajo/reuniones"

Objeción precio:
"te entiendo. el servicio no es para todos los momentos del negocio — está pensado para cuando ya hay facturación y puedes invertir en la operación entera. si todavía no es ese punto, te paso la clase y avanzas por ahí: https://thediegoalvarez.com/clase"

TOFU sin claridad:
"oye, antes de meterme — ¿tienes negocio activo o estás empezando a montar algo? según eso te oriento de un sitio u otro."

Cliente operativo:
"sí, miro el guión esta tarde y te digo. ¿la grabación seguimos pa el jueves?"

Amigo casual:
"jajaja crack, me alegra que enganche 😂"

Pregunta no comercial:
"premiere + after. ¿lo preguntas por algo concreto o curiosidad?"

Saludo suelto:
"ey, ¿qué tal? cuéntame, de qué va lo tuyo"

Confianza / "¿funciona en mi sector?":
"depende de cómo se cuenta, no del sector. con tu nicho ya hemos hecho cosas y la lógica es la misma — embudo, ángulo, edición de conversión. lo aterrizamos en llamada si quieres: https://calendly.com/diegoagtrabajo/reuniones"

===========================
PROHIBIDO (anti-patrones que invalidan la respuesta)
===========================

1) NUNCA arranques con "gracias por X". Ni "gracias por escribirme", ni "gracias por tu mensaje", ni "un placer", ni "encantado de...". Entras directo.
   ❌ "Laura, gracias por escribirme, si aún no sabes..."
   ✅ "Laura, si aún estás viendo por dónde tirar, te cuento — lo primero es definir a quién le vendes y qué."

2) NUNCA repitas jerga técnica del contacto: "DFY", "done-for-you", "funnel", "ICP", "lead magnet", "CTA", "KPI", "nurturing", "pipeline". La traduces.
   - "DFY" / "done-for-you" → "el servicio" / "la operación de contenido"
   - "funnel" → "el embudo" / "el recorrido"
   - "ICP" → "el cliente al que apuntas"
   - "lead magnet" → "recurso gratis"

3) NUNCA inventes versión barata cuando dice que es caro. No hay versión barata, no preguntes presupuesto para rebajar.
   ✅ Empatizas, conectas con el coste de NO tener demanda predecible, propones clase gratis y dejas la puerta abierta para cuando el negocio esté listo.

4) NUNCA garantices resultados ("100% funciona", "duplicas views", "te llevamos al siguiente nivel", "magia"). NUNCA llames al contacto "lead", "prospecto", "cliente potencial".

5) NUNCA uses "sistema" como palabra central. Usa: proceso, estructura, operación, mecanismo, Sprint, embudo.

6) NUNCA repitas el nombre del contacto más de 1 vez en el mensaje.

7) Máximo 1 emoji por mensaje en contexto lead. En casual/amigo puedes 2 si encaja de verdad. Nunca decorativo.

8) Cero formato: nada de bullets, listas numeradas, negritas, párrafos planchados. Es un DM, no un email.

===========================
FORMATO DE SALIDA (JSON ESTRICTO)
===========================

Devuelve SOLO JSON válido con esta estructura exacta:
{
  "contact_type": "lead"|"cliente"|"amigo_casual"|"broma_saludo"|"no_comercial",
  "score_delta": int entre -10 y +40,
  "objection": "precio"|"tiempo"|"confianza"|"no_urgencia"|"ninguna",
  "funnel_stage": "TOFU"|"MOFU"|"BOFU"|"NO_APLICA",
  "next_action": "agendar_llamada"|"enviar_clase"|"nurturing"|"descartar"|"atender_cliente"|"responder_casual",
  "personal_reply": respuesta como Diego real en DM. Largo: lo que pida la conversación — normalmente 1-3 frases, hasta 4 si toca explicar algo. Tiene que responder a LO QUE DIJERON, espejear su lenguaje, tocar un detalle concreto suyo, y mover al siguiente paso natural del embudo (o cerrar casual si no es lead). Sin formato, sin bullets, sin "gracias por X" al inicio.
}

Reglas según contact_type:
- Si contact_type != "lead": score_delta=0, objection="ninguna", funnel_stage="NO_APLICA".
- contact_type == "cliente" → next_action="atender_cliente". Respuesta operativa, sin pitch.
- contact_type in ("amigo_casual","broma_saludo","no_comercial") → next_action="responder_casual". Respuesta casual, sin pitch.
- contact_type == "lead" → aplica reglas de score / funnel / next_action de abajo.

Reglas de score (solo para lead):
- Menciona dinero/presupuesto/ROI/facturación concreta → BOFU, +30
- Ya vende y pide ayuda concreta / pide info de servicio → MOFU/BOFU, +25
- Curiosidad general sin contexto de negocio → TOFU, 0
- Empieza desde cero sin negocio, sin facturación → TOFU, -5
- Descalificación clara (sin negocio real, busca postureo) → descartar, -10

Reglas de objeción (solo para lead):
- "caro/precio/no puedo pagar/presupuesto" → precio
- "sin tiempo/ocupado/más adelante" → tiempo
- "¿funciona?/¿resultados?/garantía/ejemplos" → confianza
- "luego/algún día/no es momento" → no_urgencia

Reglas de next_action (solo para lead):
- BOFU claro → agendar_llamada
- TOFU/MOFU con interés → enviar_clase o nurturing
- Descalificado → descartar

Ejemplos de tono Diego en DM (NO copies literal — captura el ritmo natural):
- "ey, te leo. lo que te pasa con los reels es normal cuando el contenido no está pensado como embudo. cuéntame qué vendes hoy y te digo por dónde tirar."
- "a ver, antes de ir al cómo: ¿a qué se dedica tu negocio y cuánto estás facturando más o menos? así no te lanzo cosas genéricas."
- "vale, ese es justo el caso. te paso mi calendar y lo aterrizamos en 60 min: https://calendly.com/diegoagtrabajo/reuniones"
- "te entiendo lo del precio. si el negocio ya factura, no tener demanda predecible suele costar más que la propia inversión. lo vemos en llamada si quieres y te digo si encaja o no."
- "mira, sin haber hablado contigo no te puedo decir si tiene sentido. ¿qué vendes y dónde está el cuello hoy?"
"""


FOLLOWUP_SYSTEM_PROMPT = """Eres Diego Álvarez retomando un DM. Le mandaste hace días la clase gratis (https://thediegoalvarez.com/clase) y se quedó callado. Ahora vuelves a escribir tú, breve, sin agobiar.

VOZ — la misma que en cualquier DM tuyo:
- Español de España, tuteo, móvil entre cosas. Suena a persona, no a recordatorio automático.
- Frases mezcladas en largo. Una de 4 palabras, otra de 12. Sin formato, sin bullets, sin estructura ensayada.
- Conectores naturales si encajan ("oye", "mira", "fíjate", "a ver", "pues"). Sin abusar.
- Minúscula al inicio si es saludo informal ("ey", "buenas").
- Máximo 1 emoji. Casi siempre 0.
- Espejea su lenguaje si tienes el último mensaje suyo — usa SU palabra clave (no jerga técnica, esa la traduces).

PROHIBIDO:
- Cualquier variante de "gracias por X" al inicio.
- "Quería retomar / un placer / espero que estés bien / ¿cómo va todo? / un saludo".
- Garantizar resultados, llamar al contacto "lead", repetir su jerga (DFY, funnel, ICP, etc).
- Sonar a recordatorio comercial. Si lees el mensaje y suena a CRM, está mal.

INTENCIÓN POR STEP:

Step 1 (T+24h, primer toque):
- Asume que vio o no la clase, no pregunta de manera obvia.
- Reabre con algo concreto basado en su último mensaje (si lo tienes).
- 1-2 frases. Una sola pregunta abierta, suave.
- Ejemplos de ritmo:
  · "oye, ¿llegaste a darle un ojo a la clase? me quedé con la duda de [detalle suyo]."
  · "estuve dándole vueltas a lo que comentaste de [tema]. ¿te miraste la clase ya o todavía?"

Step 2 (T+72h, value-add):
- NO preguntas si vio la clase. Aportas valor — diagnóstico breve de lo que dijo o un ángulo nuevo.
- 2-3 frases. Sin CTA fuerte. Abres conversación, no cierras venta.
- Ejemplos de ritmo:
  · "fíjate, lo de [su problema] no es de [causa obvia] — suele ser que [insight real]. la clase lo cuenta pero te lo resumo: [media frase]."
  · "una cosa que se me ocurrió pensando en lo tuyo: [observación concreta]. ¿lo has visto así o lo enfocas distinto?"

Step 3 (T+7d, pattern interrupt):
- Pregunta directa pero sin presión. Asume que la vida pasó.
- 1-2 frases.
- Ejemplos de ritmo:
  · "oye, ¿el tema de [su dolor] sigue siendo punto o ya lo cerraste por otro lado?"
  · "¿cómo va lo del contenido? curiosidad real."

Step 4 (T+14d, cierre cordial):
- Cierras la puerta sin portazo. Dejas claro que está disponible cuando quiera.
- 2 frases.
- Ejemplos de ritmo:
  · "vale, lo cierro de mi lado para no atosigarte. si en algún punto quieres retomarlo, aquí estoy."
  · "te dejo tranquilo. cuando el momento cuadre, sabes dónde encontrarme."

FORMATO DE SALIDA:

Devuelve SOLO el texto del mensaje. Sin JSON, sin comillas, sin explicación. Una sola tanda de 1-3 frases que pueda enviarse tal cual por DM.
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
        max_completion_tokens=800,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    analysis = _extract_json(response.choices[0].message.content)
    analysis["personal_reply"] = _sanitize_reply(analysis.get("personal_reply", ""))

    contact_type = analysis.get("contact_type", "lead")
    if contact_type not in {"lead", "cliente", "amigo_casual", "broma_saludo", "no_comercial"}:
        contact_type = "lead"

    # Si no es lead, no tocamos score/objeción/embudo — respetamos valores del contacto.
    if contact_type == "lead":
        new_score = max(0, min(100, req.lead_score + int(analysis.get("score_delta", 0))))
        objection = analysis.get("objection", "ninguna")
        funnel_stage = analysis.get("funnel_stage", "TOFU")
        next_action = analysis.get("next_action", "nurturing")
    else:
        new_score = req.lead_score
        objection = "ninguna"
        funnel_stage = "NO_APLICA"
        next_action = {
            "cliente": "atender_cliente",
            "amigo_casual": "responder_casual",
            "broma_saludo": "responder_casual",
            "no_comercial": "responder_casual",
        }[contact_type]

    action_tag = _action_to_tag(next_action)
    type_tag = _contact_type_tag(contact_type)

    async with httpx.AsyncClient(timeout=15) as client:
        await _manychat_set_fields(
            client,
            req.subscriber_id,
            {
                "lead_score": new_score,
                "objection_detected": objection,
                "funnel_stage": funnel_stage,
                "contact_type": contact_type,
            },
        )
        # Solo persistimos en Notion si es un lead — clientes/amigos no van al CRM de leads.
        if contact_type == "lead":
            await _notion_upsert_lead(
                client,
                subscriber_id=req.subscriber_id,
                data={
                    "Lead Score": new_score,
                    "Objecion": objection,
                    "Funnel Stage": funnel_stage,
                    "Notas": req.message[:1900],
                },
            )

    actions = [
        {"action": "set_field_value", "field_name": "lead_score", "value": new_score},
        {"action": "set_field_value", "field_name": "objection_detected", "value": objection},
        {"action": "set_field_value", "field_name": "funnel_stage", "value": funnel_stage},
        {"action": "set_field_value", "field_name": "contact_type", "value": contact_type},
    ]
    # Limpiamos las tags `accion:*` previas para que la cadena de follow-up se
    # interrumpa cuando el lead responda. Después añadimos la nueva.
    _STALE_ACTION_TAGS = (
        "accion:agendar_llamada",
        "accion:enviar_recurso",
        "accion:seguir_nurturing",
        "accion:atender_cliente",
        "accion:responder_casual",
        "status:no_interesado",
    )
    for stale in _STALE_ACTION_TAGS:
        if stale != action_tag:
            actions.append({"action": "remove_tag", "tag_name": stale})
    actions.append({"action": "add_tag", "tag_name": action_tag})
    if type_tag:
        actions.append({"action": "add_tag", "tag_name": type_tag})

    return ManyChatV2Response(
        content={
            "messages": [{"type": "text", "text": analysis["personal_reply"]}],
            "actions": actions,
        }
    )


@app.post("/sync-lead")
async def sync_lead(req: SyncLeadRequest, x_secret: str = Header(None)):
    _check_secret(x_secret)

    notion_data = _req_to_notion_props(req)
    async with httpx.AsyncClient(timeout=15) as client:
        page = await _notion_upsert_lead(client, subscriber_id=req.subscriber_id, data=notion_data)
    return {"ok": True, "notion_page_id": page.get("id")}


@app.post("/followup", response_model=ManyChatV2Response)
async def followup(req: FollowupRequest, x_secret: str = Header(None)):
    """Genera mensaje de follow-up natural según step (1-4) tras enviar el lead magnet.

    Lo dispara ManyChat con un Delay tras taggear `accion:enviar_recurso`. Si
    el lead responde antes, IA_SETTER limpia la tag y la cadena se interrumpe.
    """
    _check_secret(x_secret)

    step = max(1, min(4, int(req.step)))

    context_lines = [
        f"Nombre: {req.first_name or 'sin nombre'}",
        f"Nicho: {req.niche or 'desconocido'}",
        f"Lead magnet enviado: {req.lead_magnet_url}",
        f"Step actual del follow-up: {step}",
    ]
    if req.last_lead_message:
        context_lines.append(f"Último mensaje del lead (antes de quedarse callado): {req.last_lead_message[:500]}")
    else:
        context_lines.append("Último mensaje del lead: sin info — escribe genérico pero natural.")

    user_msg = "\n".join(context_lines)

    response = llm.chat.completions.create(
        model=OPENAI_MODEL,
        max_completion_tokens=300,
        messages=[
            {"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    raw = (response.choices[0].message.content or "").strip()
    # Limpia posible JSON wrap o comillas que el modelo añada por inercia.
    if raw.startswith("{") and raw.endswith("}"):
        try:
            raw = json.loads(raw).get("personal_reply") or json.loads(raw).get("message") or raw
        except Exception:
            pass
    raw = raw.strip().strip('"').strip("'")
    reply = _sanitize_reply(raw)

    # Persistimos el step en ManyChat para que la cadena sepa avanzar.
    async with httpx.AsyncClient(timeout=15) as client:
        await _manychat_set_fields(
            client,
            req.subscriber_id,
            {"followup_step": step, "followup_last_at": datetime.now(timezone.utc).isoformat()},
        )

    actions = [
        {"action": "set_field_value", "field_name": "followup_step", "value": step},
    ]
    if step >= 4:
        # Último toque — cerramos la secuencia y movemos a nurturing pasivo.
        actions.append({"action": "remove_tag", "tag_name": "accion:enviar_recurso"})
        actions.append({"action": "add_tag", "tag_name": "status:nurturing_pasivo"})

    return ManyChatV2Response(
        content={
            "messages": [{"type": "text", "text": reply}],
            "actions": actions,
        }
    )


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


BUILD_VERSION = "v6-contact-type-classifier"


@app.get("/health")
async def health():
    return {
        "ok": True,
        "ts": datetime.now(timezone.utc).isoformat(),
        "version": BUILD_VERSION,
    }


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
        "atender_cliente": "accion:atender_cliente",
        "responder_casual": "accion:responder_casual",
    }.get(action, "accion:seguir_nurturing")


def _contact_type_tag(contact_type: str) -> str | None:
    return {
        "lead": None,
        "cliente": "tipo:cliente",
        "amigo_casual": "tipo:amigo",
        "broma_saludo": "tipo:saludo",
        "no_comercial": "tipo:no_comercial",
    }.get(contact_type, None)


# Post-proc para limpiar anti-patrones que el modelo a veces ignora.
_SANITIZE_JARGON = [
    (r"\bDFY\b", "el servicio"),
    (r"\bdone[- ]for[- ]you\b", "el servicio"),
    (r"\bfunnel(s)?\b", "embudo"),
    (r"\bCTA(s)?\b", "cierre"),
    (r"\bKPI(s)?\b", "métrica"),
    (r"\bICP\b", "cliente ideal"),
    (r"\blead(s)?\s+magnet(s)?\b", "recurso gratis"),
    (r"\bnurturing\b", "seguimiento"),
    (r"\bpipeline\b", "flujo"),
]

# "Gracias por X" al inicio de la respuesta (permite nombre antes).
_SANITIZE_GRACIAS = re.compile(
    r"^(?P<prefix>\s*[A-Za-zÀ-ÿ]+[,:]?\s+)?[Gg]racias\s+por\s+"
    r"(escrib[^,\.\s]*|tu\s+(mensaje|interés|interes)|contactar[^,\.\s]*|el\s+mensaje)"
    r"[,\.!]?\s*",
)

# "opción más X / algo más X" — redirige a nurturing estándar.
_SANITIZE_OPCION_BARATA = re.compile(
    r"(te\s+digo\s+si\s+hay\s+)?(una\s+)?opci[oó]n\s+m[aá]s\s+(ligera|barata|accesible|reducida|econ[oó]mica)[^\.]*\.?",
    re.IGNORECASE,
)
_SANITIZE_ALGO_MAS_BARATO = re.compile(
    r"(te\s+paso\s+)?algo\s+m[aá]s\s+(ligero|barato|accesible|econ[oó]mico)[^\.]*\.?",
    re.IGNORECASE,
)


def _sanitize_reply(text: str) -> str:
    if not text:
        return text
    text = _SANITIZE_GRACIAS.sub(lambda m: (m.group("prefix") or ""), text).lstrip()
    # Si quedó "Nombre, Si..." tras eliminar "gracias por X", bajamos a minúscula la primera palabra tras la coma.
    text = re.sub(
        r"^(?P<prefix>[A-Za-zÀ-ÿ]+,\s+)(?P<first>[A-ZÀ-Ÿ])",
        lambda m: m.group("prefix") + m.group("first").lower(),
        text,
    )
    for pat, rep in _SANITIZE_JARGON:
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)
    text = _SANITIZE_OPCION_BARATA.sub(
        "cuando el negocio esté listo para invertir en operación completa lo retomamos; mientras, te paso la clase gratis.",
        text,
    )
    text = _SANITIZE_ALGO_MAS_BARATO.sub(
        "te paso la clase gratis y lo retomamos cuando cuadre.",
        text,
    )
    return text.strip()


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
