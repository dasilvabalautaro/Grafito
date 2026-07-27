# Plan de entrenamiento v3 (borrador)

Fecha: 2026-07-27. Estado: **borrador** — condicionado a la experiencia de uso del demo con v2.

Este documento esboza qué supondría un tercer entrenamiento, a la luz de los
defectos que v2 dejó vivos (ver `docs/EVALUATION.md`, sección v2). No es un
plan aprobado: se activa solo si el uso real del demo confirma que los
defectos restantes lo justifican.

---

## 1. Defectos objetivo (heredados de v2)

1. **Manchas de color esporádicas en esquinas.** El filtro estricto de v2
   eliminó 44 casos seguros (recall parcial); variantes de la tira de
   calibración siguen en el dataset.
2. **Adherencia fina al prompt irregular** (ej.: `make the background blue`
   tiñó la camiseta en 2/2 seeds). CLIP plano respecto al base
   (0.2509 vs 0.2523).
3. **LPIPS sin cumplir meta** (0.2405 a 512; objetivo del plan v2: 0.18).

## 2. Mitigaciones gratuitas (implementadas primero, 2026-07-27)

Antes de gastar en entrenamiento, se implementaron en `scripts/demo.py`:

- Recorte automático de bordes (8 px por lado, conmutador activado por
  defecto) contra las manchas de esquina.
- Plantillas de prompt con atributos concretos (color/material explícito) —
  aprendizaje de v2: `add a black hat` >> `add a hat`.
- Multi-variante (1–3 seeds por edición) para elegir la mejor.

Nota técnica: la multi-variante a 512 px en MPS exige
`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` (ya fijado en `scripts/demo.py`);
con el límite por defecto, la segunda variante hace OOM aunque la primera
encaje. Validado el 2026-07-27: 2 variantes a 512 px sin OOM, ~31 s cada una.

**Criterio de activación de v3:** si con estas mitigaciones los defectos
siguen molestando en uso real (anotar casos en `docs/NEXT_DEMO.md`), se
procede con lo siguiente.

## 3. Datos (local, ~2–3 h, $0)

- **Segunda ronda del detector de tiras:** barrido de alta recall + revisión
  visual de candidatos en láminas (sin falsos positivos por alfombras/maletas
  como en v2). Re-auditoría completa.
- **Filtro de pares ruidosos:** instrucciones que no corresponden al cambio
  visual (LPIPS origen/destino + análisis del prompt). Ataca la adherencia.
- **Opcional:** dataset dedicado de edición facial para retratos (hoy solo
  sobremuestreo ×2 por keywords).

## 4. Entrenamiento (pod, ~5–7 h, ~$10–15, protocolo v2)

- Punto de partida: **v2** (currículo; v2 partió del base para no arrastrar
  defectos de v1, y funcionó).
- 8000–10000 pasos a 512 px (v1 usó 10000 a 256; v2 se quedó en 6000 por
  presupuesto).
- Mismo protocolo anti-costes: imagen verificada con pre-flight, sin Network
  Volume, presupuesto con alarma, destrucción el mismo día.

## 5. Evaluación (deuda técnica de v2, cerrarla)

- Subir **v1 al pod desde Drive** (Drive→pod es rápido; desde casa no).
- Una sola corrida del arnés: base vs v1 vs v2 vs v3, **los cuatro a 512 px**,
  mismos 528 ejemplos. Cierra la comparación manzana-a-manzana que quedó
  abierta en v2 (v1 nunca se midió a 512).
- Panel cualitativo idéntico al de v2 para comparabilidad, más los casos que
  fallen en el uso real del demo.

## 6. Criterios de adopción (mismos que v2, ajustados)

- LPIPS ≤ 0.18 a 512 **con v1 medido a 512 en la misma corrida**.
- CLIP ≥ 0.25 sin regresión.
- Panel ≥ 4/5 en caras **y** ≥ 4/5 en adherencia fina (casos del demo).
- Si no se cumplen: se conserva v2 y el coste queda acotado a ~$15.
